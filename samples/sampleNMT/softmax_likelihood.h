#ifndef SAMPLE_NMT_SOFTMAX_LIKELIHOOD_
#define SAMPLE_NMT_SOFTMAX_LIKELIHOOD_

#include "NvInfer.h"
#include "likelihood.h"

namespace nmtSample
{
    /** \class SoftmaxLikelihood
    *
    * \brief calculates softmax likelihood and TopK indices for the raw input logits
    *
    */
    class SoftmaxLikelihood : public Likelihood
    {
    private:
        class SoftmaxLikelihoodCombinationOperator : public LikelihoodCombinationOperator
        {
        public:
            SoftmaxLikelihoodCombinationOperator() = default;

            virtual float combine(float rayLikelihood, float optionLikelihood) const;

            virtual float init() const;

            virtual ~SoftmaxLikelihoodCombinationOperator() = default;
        };

    public:
        SoftmaxLikelihood() = default;

        virtual LikelihoodCombinationOperator::ptr getLikelihoodCombinationOperator() const;

        virtual void addToModel(
            nvinfer1::INetworkDefinition * network,
            int beamWidth,
            nvinfer1::ITensor * inputLogits,
            nvinfer1::ITensor ** newLikelihoods,
            nvinfer1::ITensor ** newVocabularyIndices);

        virtual std::string getInfo();

        virtual ~SoftmaxLikelihood() = default;
    };
}

#endif // SAMPLE_NMT_SOFTMAX_LIKELIHOOD_
