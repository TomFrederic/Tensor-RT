#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

const int poolingH = 6;
const int poolingW = 6;
const int featureStride = 16;
const int preNmsTop = 6000;
const int nmsMaxOut = 300;
const int anchorsRatioCount = 3;
const int anchorsScaleCount = 3;
const float iouThreshold = 0.7f;
const float minBoxSize = 16;
const float spatialScale = 0.03125f;
const float anchorsRatios[anchorsRatioCount] = { 0.5f, 1.0f, 2.0f };
const float anchorsScales[anchorsScaleCount] = { 8.0f, 16.0f, 32.0f };

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
class FRCNNPluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
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
