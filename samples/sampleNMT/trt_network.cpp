#include "trt_network.h"

namespace nmtSample
{
    TRTNetwork::TRTNetwork(nvinfer1::INetworkDefinition * net)
        : mNetwork(net)
    {
    }

    TRTNetwork::~TRTNetwork()
    {
        if (mNetwork)
        {
            mNetwork->destroy();
            mNetwork = nullptr;
        }
    }

    nvinfer1::INetworkDefinition* TRTNetwork::getImpl() const
    {
        return mNetwork;
    }
}
