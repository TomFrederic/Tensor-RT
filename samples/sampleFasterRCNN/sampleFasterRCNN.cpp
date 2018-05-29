#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <cstring>
#include <algorithm>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <sys/time.h>
#include <queue>
#include <pthread.h>
#include <unistd.h>
#include <omp.h>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "common.h"


static Logger gLogger;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using namespace cv;

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_C = 3;
//static const int INPUT_H = 375;
//static const int INPUT_W = 500;
static const int INPUT_H = 720;
static const int INPUT_W = 1280;
static const int IM_INFO_SIZE = 3;
static const int OUTPUT_CLS_SIZE = 21;
static const int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;

const std::string CLASSES[OUTPUT_CLS_SIZE]{ "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

const char* INPUT_BLOB_NAME0 = "data";
const char* INPUT_BLOB_NAME1 = "im_info";
const char* OUTPUT_BLOB_NAME0 = "bbox_pred";
const char* OUTPUT_BLOB_NAME1 = "cls_prob";
const char* OUTPUT_BLOB_NAME2 = "rois";

#define USE_MODE0 0

#if USE_MODE0
const int poolingH = 7;
const int poolingW = 7;
const float spatialScale = 0.0625f;
#else
const int poolingH = 6;
const int poolingW = 6;
const float spatialScale = 0.03125f;
#endif

const int featureStride = 16;
const int preNmsTop = 6000;
const int nmsMaxOut = 300;
const int anchorsRatioCount = 3;
const int anchorsScaleCount = 3;
const float iouThreshold = 0.7f;
const float minBoxSize = 16;

const float anchorsRatios[anchorsRatioCount] = { 0.5f, 1.0f, 2.0f };
const float anchorsScales[anchorsScaleCount] = { 8.0f, 16.0f, 32.0f };

long GetTickCount()
{
	struct timeval tv;
	gettimeofday(&tv,NULL);
	return (tv.tv_sec*1000000+tv.tv_usec)/1000;
}

struct PPM
{
	std::string magic, fileName;
	int h, w, max;
	uint8_t buffer[INPUT_C*INPUT_H*INPUT_W];
};

struct BBox
{
	float x1, y1, x2, y2;
};

struct DetectOutput
{
	BBox box;
	float score;
	std::string name;
};

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/samples/faster-rcnn/", "data/faster-rcnn/"};
    return locateFile(input, dirs);
}

// simple PPM (portable pixel map) reader
void readPPMFile(const std::string& filename, PPM& ppm)
{
	ppm.fileName = filename;
	std::ifstream infile(locateFile(filename), std::ifstream::binary);
	infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
	infile.seekg(1, infile.cur);
	infile.read(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

void writePPMFileWithBBox(const std::string& filename, PPM& ppm, const BBox& bbox)
{
	std::ofstream outfile("./" + filename, std::ofstream::binary);
	assert(!outfile.fail());
	outfile << "P6" << "\n" << ppm.w << " " << ppm.h << "\n" << ppm.max << "\n";
	auto round = [](float x)->int {return int(std::floor(x + 0.5f)); };
	for (int x = int(bbox.x1); x < int(bbox.x2); ++x)
	{
		// bbox top border
		ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3] = 255;
		ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3 + 1] = 0;
		ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3 + 2] = 0;
		// bbox bottom border
		ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3] = 255;
		ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3 + 1] = 0;
		ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3 + 2] = 0;
	}
	for (int y = int(bbox.y1); y < int(bbox.y2); ++y)
	{
		// bbox left border
		ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3] = 255;
		ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3 + 1] = 0;
		ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3 + 2] = 0;
		// bbox right border
		ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3] = 255;
		ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3 + 1] = 0;
		ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3 + 2] = 0;
	}
	outfile.write(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}
	
void caffeToGIEModel(const std::string& deployFile,			// name for caffe prototxt
	const std::string& modelFile,			// name for model 
	const std::vector<std::string>& outputs,		// network outputs
	unsigned int maxBatchSize,				// batch size - NB must be at least as large as the batch we want to run with)
	nvcaffeparser1::IPluginFactory* pluginFactory,	// factory for plugin layers
	IHostMemory **gieModelStream)			// output stream for the GIE model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	
	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	parser->setPluginFactory(pluginFactory);

	std::cout << "Begin parsing model..." << std::endl;
    bool fp16 = builder->platformHasFastFp16();

	const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
		locateFile(modelFile).c_str(),
		*network,
		nvinfer1::DataType::kHALF);
	std::cout << "End parsing model..." << std::endl;
	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1024 << 20);	// we need about 6MB of scratch space for the plugin layer for batch size 5
    builder->setHalf2Mode(true);

	std::cout << "Begin building engine..." << std::endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);
	std::cout << "End building engine..." << std::endl;

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	(*gieModelStream) = engine->serialize();

	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void doInference(IExecutionContext* context, float* inputData, float* inputImInfo, float* outputBboxPred, float* outputClsProb, float *outputRois, int batchSize)
{
	const ICudaEngine& engine = context->getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly 2 inputs and 3 outputs.
	assert(engine.getNbBindings() == 5);
	void* buffers[5];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex0 = engine.getBindingIndex(INPUT_BLOB_NAME0),
	inputIndex1 = engine.getBindingIndex(INPUT_BLOB_NAME1),
	outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
	outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1),
	outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2);


	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex0], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));   // data
	CHECK(cudaMalloc(&buffers[inputIndex1], batchSize * IM_INFO_SIZE * sizeof(float)));                  // im_info
	CHECK(cudaMalloc(&buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float))); // bbox_pred
	CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float)));  // cls_prob
	CHECK(cudaMalloc(&buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float)));                // rois

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	cudaEvent_t start, end;
	CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
	CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

	float total = 0, ms=0;
	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex0], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));

	CHECK(cudaMemcpyAsync(buffers[inputIndex1], inputImInfo, batchSize * IM_INFO_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));

	cudaEventRecord(start, stream);
	context->enqueue(batchSize, buffers, stream, nullptr);
	cudaEventRecord(end, stream);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&ms, start, end);
	total += ms;
	std::cout << " runs  " << total << " ms." << std::endl;

	CHECK(cudaMemcpyAsync(outputBboxPred, buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(outputClsProb, buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(outputRois, buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex0]));
	CHECK(cudaFree(buffers[inputIndex1]));
	CHECK(cudaFree(buffers[outputIndex0]));
	CHECK(cudaFree(buffers[outputIndex1]));
	CHECK(cudaFree(buffers[outputIndex2]));
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

template<int OutC>
class Reshape : public IPlugin
{
public:
	Reshape() {}
	Reshape(const void* buffer, size_t size)
	{
		assert(size == sizeof(mCopySize));
		mCopySize = *reinterpret_cast<const size_t*>(buffer);
	}

	int getNbOutputs() const override
	{
		return 1;
	}
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		assert(nbInputDims == 1);
		assert(index == 0);
		assert(inputs[index].nbDims == 3);
		assert((inputs[0].d[0])*(inputs[0].d[1]) % OutC == 0);
		return DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] / OutC, inputs[0].d[2]);
	}

	int initialize() override
	{
		return 0;
	}

	void terminate() override
	{
	}

	size_t getWorkspaceSize(int) const override
	{
		return 0;
	}

	// currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
	int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
	{
		CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
		return 0;
	}

	
	size_t getSerializationSize() override
	{
		return sizeof(mCopySize);
	}

	void serialize(void* buffer) override
	{
		*reinterpret_cast<size_t*>(buffer) = mCopySize;
	}

	void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
	{
		mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
	}

protected:
	size_t mCopySize;
};


// integration for serialization
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
	// deserialization plugin implementation
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
	{
		assert(isPlugin(layerName));
		if (!strcmp(layerName, "ReshapeCTo2"))
		{
			assert(mPluginRshp2 == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginRshp2 = std::unique_ptr<Reshape<2>>(new Reshape<2>());
			return mPluginRshp2.get();
		}
		else if (!strcmp(layerName, "ReshapeCTo18"))
		{
			assert(mPluginRshp18 == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginRshp18 = std::unique_ptr<Reshape<18>>(new Reshape<18>());
			return mPluginRshp18.get();
		}
		else if (!strcmp(layerName, "RPROIFused"))
		{
			assert(mPluginRPROI == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
				(createFasterRCNNPlugin(featureStride, preNmsTop, nmsMaxOut, iouThreshold, minBoxSize, spatialScale,
					DimsHW(poolingH, poolingW), Weights{ nvinfer1::DataType::kFLOAT, anchorsRatios, anchorsRatioCount },
					Weights{ nvinfer1::DataType::kFLOAT, anchorsScales, anchorsScaleCount }), nvPluginDeleter);
			return mPluginRPROI.get();
		}
		else
		{
			assert(0);
			return nullptr;
		}
	}

	IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{
		assert(isPlugin(layerName));
		if (!strcmp(layerName, "ReshapeCTo2"))
		{
			assert(mPluginRshp2 == nullptr);
			mPluginRshp2 = std::unique_ptr<Reshape<2>>(new Reshape<2>(serialData, serialLength));
			return mPluginRshp2.get();
		}
		else if (!strcmp(layerName, "ReshapeCTo18"))
		{
			assert(mPluginRshp18 == nullptr);
			mPluginRshp18 = std::unique_ptr<Reshape<18>>(new Reshape<18>(serialData, serialLength));
			return mPluginRshp18.get();
		}
		else if (!strcmp(layerName, "RPROIFused"))
		{
			assert(mPluginRPROI == nullptr);
			mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createFasterRCNNPlugin(serialData, serialLength), nvPluginDeleter);
			return mPluginRPROI.get();
		}
		else
		{
			assert(0);
			return nullptr;
		}
	}

	// caffe parser plugin implementation
	bool isPlugin(const char* name) override
	{
		return (!strcmp(name, "ReshapeCTo2")
			|| !strcmp(name, "ReshapeCTo18")
			|| !strcmp(name, "RPROIFused"));
	}

	// the application has to destroy the plugin when it knows it's safe to do so
	void destroyPlugin()
	{
		mPluginRshp2.release();		mPluginRshp2 = nullptr;
		mPluginRshp18.release();	mPluginRshp18 = nullptr;
		mPluginRPROI.release();		mPluginRPROI = nullptr;
	}


	std::unique_ptr<Reshape<2>> mPluginRshp2{ nullptr };
	std::unique_ptr<Reshape<18>> mPluginRshp18{ nullptr };
	void(*nvPluginDeleter)(INvPlugin*) { [](INvPlugin* ptr) {ptr->destroy(); } };
	std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mPluginRPROI{ nullptr, nvPluginDeleter };
};


void bboxTransformInvAndClip(float* rois, float* deltas, float* predBBoxes, float* imInfo,
	const int N, const int nmsMaxOut, const int numCls)
{
	float width, height, ctr_x, ctr_y;
	float dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
	float *deltas_offset, *predBBoxes_offset, *imInfo_offset;
	for (int i = 0; i < N * nmsMaxOut; ++i)
	{
		width = rois[i * 4 + 2] - rois[i * 4] + 1;
		height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
		ctr_x = rois[i * 4] + 0.5f * width;
		ctr_y = rois[i * 4 + 1] + 0.5f * height;
		deltas_offset = deltas + i * numCls * 4;
		predBBoxes_offset = predBBoxes + i * numCls * 4;
		imInfo_offset = imInfo + i / nmsMaxOut * 3;
		for (int j = 0; j < numCls; ++j)
		{
			dx = deltas_offset[j * 4];
			dy = deltas_offset[j * 4 + 1];
			dw = deltas_offset[j * 4 + 2];
			dh = deltas_offset[j * 4 + 3];
			pred_ctr_x = dx * width + ctr_x;
			pred_ctr_y = dy * height + ctr_y;
			pred_w = exp(dw) * width;
			pred_h = exp(dh) * height;
			predBBoxes_offset[j * 4] = std::max(std::min(pred_ctr_x - 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 1] = std::max(std::min(pred_ctr_y - 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 2] = std::max(std::min(pred_ctr_x + 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 3] = std::max(std::min(pred_ctr_y + 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
		}
	}
}

std::vector<int> nms(std::vector<std::pair<float, int> >& score_index, float* bbox, const int classNum, const int numClasses, const float nms_threshold)
{
	auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
		if (x1min > x2min) {
			std::swap(x1min, x2min);
			std::swap(x1max, x2max);
		}
		return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
	};
	auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
		float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
		float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
		float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
		float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
		float overlap2D = overlapX * overlapY;
		float u = area1 + area2 - overlap2D;
		return u == 0 ? 0 : overlap2D / u;
	};

	std::vector<int> indices;
	for (auto i : score_index)
	{
		const int idx = i.second;
		bool keep = true;
		for (unsigned k = 0; k < indices.size(); ++k)
		{
			if (keep)
			{
				const int kept_idx = indices[k];
				float overlap = computeIoU(&bbox[(idx*numClasses + classNum) * 4],
					&bbox[(kept_idx*numClasses + classNum) * 4]);
				keep = overlap <= nms_threshold;
			}
			else
				break;
		}
		if (keep) indices.push_back(idx);
	}
	return indices;
}

int doInferenceWapper(IExecutionContext *context,cv::Mat &img,const int N,std::vector<DetectOutput> & outputs)
{
	std::vector<PPM> ppms(N);
	float* data = new float[N*INPUT_C*INPUT_H*INPUT_W];

	cv::Mat sample;
	cv::cvtColor(img, sample, cv::COLOR_RGB2BGR);

	float imInfo[N * 3]; // input im_info	

	long t0=GetTickCount();	

	//#pragma omp parallel for
	for (int i = 0; i < N; ++i)
	{
		imInfo[i * 3] = float(img.rows);   // number of rows
		imInfo[i * 3 + 1] = float(img.cols); // number of columns
		imInfo[i * 3 + 2] = 1;         // image scale

		int width = img.cols;
		int height = img.rows;
		float* input_data = &data[i*INPUT_C*INPUT_H*INPUT_W];
		vector<cv::Mat> input_channels;

		for(int i=0;i<INPUT_C;i++)
		{
		    cv::Mat channel(height, width, CV_32FC1, input_data);
		    input_channels.push_back(channel);
		    input_data += height*width;
		}

		cv::Mat mean_(img.rows, img.cols, CV_32FC3, cv::Scalar(102.9801f, 115.9465f, 122.7717f));

		cv::Mat sample_float;
		sample.convertTo(sample_float, CV_32FC3);
		cv::Mat sample_normalized;

		cv::subtract(sample_float, mean_, sample_normalized);
		cv::split(sample_normalized,input_channels);
	}

	long t1=GetTickCount();
	printf("pre process usetime:%d\n",t1-t0);

	// host memory for outputs 
	float* rois = new float[N * nmsMaxOut * 4];
	float* bboxPreds = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];
	float* clsProbs = new float[N * nmsMaxOut * OUTPUT_CLS_SIZE];

	// predicted bounding boxes
	float* predBBoxes = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];

		// run inference
	doInference(context, data, imInfo, bboxPreds, clsProbs, rois, N);
	

	// unscale back to raw image space
	for (int i = 0; i < N; ++i)
	{
		float * rois_offset = rois + i * nmsMaxOut * 4;
		for (int j = 0; j < nmsMaxOut * 4 && imInfo[i * 3 + 2] != 1; ++j)
			rois_offset[j] /= imInfo[i * 3 + 2];
	}

	bboxTransformInvAndClip(rois, bboxPreds, predBBoxes, imInfo, N, nmsMaxOut, OUTPUT_CLS_SIZE);

	const float nms_threshold = 0.3f;
	const float score_threshold = 0.8f;

	for (int i = 0; i < N; ++i)
	{
		float *bbox = predBBoxes + i * nmsMaxOut * OUTPUT_BBOX_SIZE;
		float *scores = clsProbs + i * nmsMaxOut * OUTPUT_CLS_SIZE;
		for (int c = 1; c < OUTPUT_CLS_SIZE; ++c) // skip the background
		{
			std::vector<std::pair<float, int> > score_index;
			for (int r = 0; r < nmsMaxOut; ++r)
			{
				if (scores[r*OUTPUT_CLS_SIZE + c] > score_threshold)
				{
					score_index.push_back(std::make_pair(scores[r*OUTPUT_CLS_SIZE + c], r));
					std::stable_sort(score_index.begin(), score_index.end(),
						[](const std::pair<float, int>& pair1,
							const std::pair<float, int>& pair2) {
						return pair1.first > pair2.first;
					});
				}
			}

			// apply NMS algorithm
			std::vector<int> indices = nms(score_index, bbox, c, OUTPUT_CLS_SIZE, nms_threshold);
			// Show results
			for (unsigned k = 0; k < indices.size(); ++k)
			{
				int idx = indices[k];
#if 0
				std::cout << "AAA==" << idx << std::endl;
				std::string storeName = CLASSES[c] + "-" + std::to_string(scores[idx*OUTPUT_CLS_SIZE + c]) + ".ppm";
				std::cout << "Detected " << CLASSES[c] << " in " << ppms[i].fileName << " with confidence " << scores[idx*OUTPUT_CLS_SIZE + c] * 100.0f << "% "
					<< " (Result stored in " << storeName << ")." << std::endl;
#endif
				BBox b{ bbox[idx*OUTPUT_BBOX_SIZE + c * 4], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 1], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 2], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 3] };
				//writePPMFileWithBBox(storeName, ppms[i], b);


				DetectOutput item;
				item.box=b;
				item.name=CLASSES[c];
				item.score=scores[idx*OUTPUT_CLS_SIZE + c];
				outputs.push_back(item);
			}
		}
	}

	delete[] data;
	delete[] rois;
	delete[] bboxPreds;
	delete[] clsProbs;
	delete[] predBBoxes;
}

#define MAX_QUEUE_SIZE 32

static queue<cv::Mat> images_queue;
static pthread_mutex_t mutex;
static int bRunning=true;

static void *video_source_thread(void *arg)
{
	cv::VideoCapture *webcam = (cv::VideoCapture *)arg;
	while (1)
	{	
		pthread_mutex_lock(&mutex);
		cv::Mat img;
		
		if(images_queue.size()<MAX_QUEUE_SIZE)
		{
		  	*webcam >> img;
			if(img.cols==0)
			{
			   printf("invalid image\n");
			   pthread_mutex_unlock(&mutex);
			   break;
			}
			images_queue.push(img.clone());
			pthread_mutex_unlock(&mutex);	
		}	
		else
		{
			pthread_mutex_unlock(&mutex);	
			usleep(1000);	
		}		
	}
	bRunning=false;
}

int main(int argc, char** argv)
{
	printf("BuildTime:%s %s\n",__DATE__,__TIME__);

	assert(argc==2);

	pthread_mutex_init(&mutex,NULL);

	string video_path = argv[1];
	cv::VideoCapture webcam = cv::VideoCapture(video_path);
	if (!webcam.isOpened())
	{
		webcam.release();
		std::cerr << "Error during opening capture device!" << std::endl;
		return 1;
	}
	else
	{
		std::cerr << "can open the video!" << std::endl;
		pthread_t pid;
		pthread_create(&pid,0,video_source_thread,&webcam);
	}

	// create a GIE model from the caffe model and serialize it to a stream
	PluginFactory pluginFactory;
	IHostMemory *gieModelStream{ nullptr };
	// batch size

	const int N = 32;
	const int screen_w=1920;
	const int screen_h=1080;
	const int cols=8;
	const int rows=4;

	const int cell_w=screen_w/cols;
	const int cell_h=screen_h/rows;
	
	cv::Size newSize;
	newSize.height=cell_h;
	newSize.width=cell_w;	

	cv::Mat screen_image(screen_h, screen_w, CV_8UC3);

#if USE_MODE0
	caffeToGIEModel("faster_rcnn_test_iplugin.prototxt",
		"VGG16_faster_rcnn_final.caffemodel",
		std::vector < std::string > { OUTPUT_BLOB_NAME0, OUTPUT_BLOB_NAME1, OUTPUT_BLOB_NAME2 },
		N, &pluginFactory, &gieModelStream);
#else
	caffeToGIEModel("faster_rcnn_test_iplugin.prototxt_xinkai",
		"VGG16_faster_rcnn_final.caffemodel_xinkai",
		std::vector < std::string > { OUTPUT_BLOB_NAME0, OUTPUT_BLOB_NAME1, OUTPUT_BLOB_NAME2 },
		N, &pluginFactory, &gieModelStream);
#endif
	pluginFactory.destroyPlugin();
	// read a random sample image
	srand(unsigned(time(nullptr)));
	// available images 

	// deserialize the engine 
	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);

	IExecutionContext *context = engine->createExecutionContext();

	//Read Video...
	int nFrames = 0;
	char s_image[30];
	int time_interval=40;
	
	long counter=0;
	long last_t=GetTickCount();
	float fps=0;
	
	cv::namedWindow("img",CV_WINDOW_NORMAL|WINDOW_OPENGL);
	setWindowProperty("img",CV_WND_PROP_FULLSCREEN,CV_WINDOW_FULLSCREEN);

	while (bRunning)
	{
		pthread_mutex_lock(&mutex);

		if(images_queue.empty())
		{
			pthread_mutex_unlock(&mutex);
			printf("decode slow\n");
			usleep(1000*40);
			continue;	
		}	

		cv::Mat img=images_queue.front().clone();
		images_queue.pop();
		pthread_mutex_unlock(&mutex);		

		nFrames++;
#if 1
		std::vector<DetectOutput> outputs;
		doInferenceWapper(context,img,N,outputs);

		for(auto a:outputs)
		{
		     stringstream ss;
		     ss<<a.name<<"@"<<a.score;
		     //cout<<ss.str();
		     Rect rect(a.box.x1,a.box.y1,a.box.x2-a.box.x1,a.box.y2-a.box.y1);
		     //cout<<rect<<endl;
		     rectangle(img,rect,Scalar(0,0,255),2);
		     putText(img,ss.str(),Point(a.box.x1,(a.box.y1+a.box.y2)/2),CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0));
		}
		
		if(counter++==30)
		{
			long this_t=GetTickCount();
			fps=30.f/(this_t-last_t);
			last_t=this_t;
			counter=0;
		}

#endif
		cv::Mat newSample;
		cv::resize(img,newSample,newSize);
		
		//Scalar value(255,0,0);
		//cv::Mat out=newSample;
		//copyMakeBorder(newSample,out,2,2,2,2,BORDER_CONSTANT,value);


		for(int i=0;i<rows;i++)
		{
			for(int j=0;j<cols;j++)
			{
			    newSample.copyTo(screen_image(cv::Rect(j*cell_w,i*cell_h,cell_w,cell_h)));
			}
		}

		stringstream ss;
		ss<<fps*1000<<" fps";
		putText(screen_image,ss.str(),Point(10,30),CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0));
		imshow("img",screen_image);
		
		waitKey(1);	
	}

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
	pluginFactory.destroyPlugin();

	return 0;
}

