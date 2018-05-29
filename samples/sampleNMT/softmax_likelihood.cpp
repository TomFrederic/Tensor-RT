#include "softmax_likelihood.h"

#include <cassert>

#include "debug_util.h"
#include <fstream>

namespace nmtSample
{
    void SoftmaxLikelihood::addToModel(
        nvinfer1::INetworkDefinition * network,
        int beamWidth,
        nvinfer1::ITensor * inputLogits,
        nvinfer1::ITensor ** newLikelihoods,
        nvinfer1::ITensor ** newVocabularyIndices)
    {
        auto softmaxLayer = network->addSoftMax(*inputLogits);
        assert(softmaxLayer != nullptr);
        softmaxLayer->setAxes(2);
        auto softmaxTensor = softmaxLayer->getOutput(0);
        assert(softmaxTensor != nullptr);

        auto topKLayer = network->addTopK(*softmaxTensor, nvinfer1::TopKOperation::kMAX, beamWidth, 2);
        assert(topKLayer != nullptr);
        *newLikelihoods = topKLayer->getOutput(0);
        assert(*newLikelihoods != nullptr);
        *newVocabularyIndices = topKLayer->getOutput(1);
        assert(*newVocabularyIndices != nullptr);
    }

    float SoftmaxLikelihood::SoftmaxLikelihoodCombinationOperator::combine(float rayLikelihood, float optionLikelihood) const
    {
        return rayLikelihood * optionLikelihood;
    }

    float SoftmaxLikelihood::SoftmaxLikelihoodCombinationOperator::init() const
    {
        return 1.0F;
    }

    LikelihoodCombinationOperator::ptr SoftmaxLikelihood::getLikelihoodCombinationOperator() const
    {
        return std::make_shared<SoftmaxLikelihoodCombinationOperator>();
    }

    std::string SoftmaxLikelihood::getInfo()
    {
        return "Softmax Likelihood";
    }
}
