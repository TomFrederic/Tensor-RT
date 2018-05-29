#ifndef SAMPLE_NMT_BLEU_SCORE_WRITER_
#define SAMPLE_NMT_BLEU_SCORE_WRITER_

#include <memory>
#include <istream>
#include <vector>

#include "data_writer.h"
#include "vocabulary.h"

namespace nmtSample
{
    /** \class BLEUScoreWriter
    *
    * \brief all it does is to evaluate BLEU score
    *
    */
    class BLEUScoreWriter : public DataWriter
    {
    public:
        BLEUScoreWriter(std::shared_ptr<std::istream> referenceTextInput,
                        Vocabulary::ptr vocabulary,
                        int maxOrder = 4);

        virtual void write(
            int sampleCount,
            int maxOutputSequenceLength,
            const int * hOutputData,
            const int * hActualOutputSequenceLengths);

        virtual void initialize();

        virtual void finalize();

        virtual std::string getInfo();

        float getScore() const;

        virtual ~BLEUScoreWriter() = default;

    private:
        std::shared_ptr<std::istream> mReferenceInput;
        Vocabulary::ptr mVocabulary;
        size_t mReferenceLength;
        size_t mTranslationLength;
        int mMaxOrder;
        bool mSmooth;
        std::vector<size_t> mMatchesByOrder;
        std::vector<size_t> mPossibleMatchesByOrder;
    };
}

#endif // SAMPLE_NMT_BLEU_SCORE_WRITER_
