#ifndef SAMPLE_NMT_TEXT_WRITER_
#define SAMPLE_NMT_TEXT_WRITER_

#include <memory>
#include <ostream>

#include "data_writer.h"
#include "vocabulary.h"

namespace nmtSample
{
    /** \class TextReader
    *
    * \brief writes sequences of data into output stream
    *
    */
    class TextWriter : public DataWriter
    {
    public:
        TextWriter(std::shared_ptr<std::ostream> textOnput, Vocabulary::ptr vocabulary);

        virtual void write(
            int sampleCount,
            int maxOutputSequenceLength,
            const int * hOutputData,
            const int * hActualOutputSequenceLengths);

        virtual void initialize();

        virtual void finalize();

        virtual std::string getInfo();

        virtual ~TextWriter() = default;

    private:
        std::shared_ptr<std::ostream> mOutput;
        Vocabulary::ptr mVocabulary;
    };
}

#endif // SAMPLE_NMT_TEXT_WRITER_
