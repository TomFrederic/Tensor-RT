#ifndef SAMPLE_NMT_TRT_NETWORK_
#define SAMPLE_NMT_TRT_NETWORK_

#include <memory>

#include "NvInfer.h"

namespace nmtSample
{
    /** \class TRTNetwork
    *
    * \brief convenient wrapper for TensorRT INetworkDefinition
    *
    */
    class TRTNetwork
    {
    public:
        typedef std::shared_ptr<TRTNetwork> ptr;

        TRTNetwork() = delete;

        TRTNetwork(nvinfer1::INetworkDefinition * net);

        virtual ~TRTNetwork();

        nvinfer1::INetworkDefinition* getImpl() const;

    private:
        nvinfer1::INetworkDefinition * mNetwork;
    };
}

#endif // SAMPLE_NMT_TRT_NETWORK_
