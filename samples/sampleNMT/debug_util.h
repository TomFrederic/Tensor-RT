#ifndef SAMPLE_NMT_DEBUG_UTIL_
#define SAMPLE_NMT_DEBUG_UTIL_

#include "NvInfer.h"

#include <memory>
#include <ostream>
#include <list>

#include "pinned_host_buffer.h"

namespace nmtSample
{
    /** \class DebugUtil
    *
    * \brief container for static debug utility functions
    *
    */
    class DebugUtil
    {
    private:
        class DumpTensorPlugin : public nvinfer1::IPlugin
        {
        public:
            typedef std::shared_ptr<DumpTensorPlugin> ptr;

            DumpTensorPlugin(std::shared_ptr<std::ostream> out);

            virtual ~DumpTensorPlugin() = default;

        	virtual int getNbOutputs() const;

        	virtual nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims * inputs, int nbInputDims);

        	virtual void configure(const nvinfer1::Dims * inputDims, int nbInputs, const nvinfer1::Dims * outputDims, int nbOutputs, int maxBatchSize);

        	virtual int initialize();

        	virtual void terminate();

        	virtual size_t getWorkspaceSize(int maxBatchSize) const;

        	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream);

        	virtual size_t getSerializationSize();

        	virtual void serialize(void* buffer);

        private:
            std::shared_ptr<std::ostream> mOut;
            nvinfer1::Dims mDims;
            int mMaxBatchSize;
            int mTensorVolume;
            int mElemsPerRow;
            PinnedHostBuffer<float>::ptr mData;
        };

    public:
        static void addDumpTensorToStream(
            nvinfer1::INetworkDefinition * network,
            nvinfer1::ITensor * input,
            nvinfer1::ITensor ** output,
            std::shared_ptr<std::ostream> out);

    private:
        static std::list<DumpTensorPlugin::ptr> mPlugins;
    };
}

#endif // SAMPLE_NMT_DEBUG_UTIL_
