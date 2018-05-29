#ifndef SAMPLE_NMT_LIMITED_SAMPLES_DATA_READER_
#define SAMPLE_NMT_LIMITED_SAMPLES_DATA_READER_

#include "data_reader.h"

namespace nmtSample
{
    /** \class LimitedSamplesDataReader
    *
    * \brief wraps another data reader and limits the number of samples to read
    *
    */
    class LimitedSamplesDataReader : public DataReader
    {
    public:
        LimitedSamplesDataReader(int maxSamplesToRead, DataReader::ptr originalDataReader);

        int read(
            int samplesToRead,
            int maxInputSequenceLength,
            int * hInputData,
            int * hActualInputSequenceLengths);

        void reset();

        virtual std::string getInfo();

    private:
        int gMaxSamplesToRead;
        DataReader::ptr gOriginalDataReader;
        int gCurrentPosition;
    };
}

#endif // SAMPLE_NMT_LIMITED_SAMPLES_DATA_READER_
