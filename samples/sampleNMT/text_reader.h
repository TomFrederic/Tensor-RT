#ifndef SAMPLE_NMT_TEXT_READER_
#define SAMPLE_NMT_TEXT_READER_

#include <memory>
#include <istream>
#include <string>
#include "data_reader.h"
#include "vocabulary.h"

namespace nmtSample
{
    /** \class TextReader
    *
    * \brief reads sequences of data from input stream
    *
    */
    class TextReader : public DataReader
    {
    public:
        TextReader(std::shared_ptr<std::istream> textInput, Vocabulary::ptr vocabulary);

        int read(
            int samplesToRead,
            int maxInputSequenceLength,
            int * hInputData,
            int * hActualInputSequenceLengths);

        void reset();

        virtual std::string getInfo();
        
    private:
        std::shared_ptr<std::istream> mInput;
        Vocabulary::ptr mVocabulary;
    };
}

#endif // SAMPLE_NMT_TEXT_READER_
