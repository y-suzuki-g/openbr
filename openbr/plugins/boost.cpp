#include "openbr_internal.h"
#include "openbr/openbr_plugin.h"
#include "openbr/core/boost.h"

#include <QtConcurrent>

using namespace cv;

namespace br
{

class BoostClassifier : public Classifier
{
    Q_OBJECT

public:
    Q_ENUMS(Type)
    enum Type { Discrete = CvBoost::DISCRETE,
                Real = CvBoost::REAL,
                Logit = CvBoost::LOGIT,
                Gentle = CvBoost::GENTLE };

    Q_PROPERTY(br::Representation* representation READ get_representation WRITE set_representation RESET reset_representation STORED false)
    Q_PROPERTY(Type boostType READ get_boostType WRITE set_boostType RESET reset_boostType STORED false)
    Q_PROPERTY(int precalcBufSize READ get_precalcBufSize WRITE set_precalcBufSize RESET reset_precalcBufSize STORED false)
    Q_PROPERTY(float minTAR READ get_minTAR WRITE set_minTAR RESET reset_minTAR STORED false)
    Q_PROPERTY(float maxFAR READ get_maxFAR WRITE set_maxFAR RESET reset_maxFAR STORED false)
    Q_PROPERTY(float trimRate READ get_trimRate WRITE set_trimRate RESET reset_trimRate STORED false)
    Q_PROPERTY(int maxDepth READ get_maxDepth WRITE set_maxDepth RESET reset_maxDepth STORED false)
    Q_PROPERTY(int maxWeakCount READ get_maxWeakCount WRITE set_maxWeakCount RESET reset_maxWeakCount STORED false)
    BR_PROPERTY(br::Representation*, representation, NULL)
    BR_PROPERTY(Type, boostType, Gentle)
    BR_PROPERTY(int, precalcBufSize, 1024)
    BR_PROPERTY(float, minTAR, 0.995)
    BR_PROPERTY(float, maxFAR, 0.5)
    BR_PROPERTY(float, trimRate, 0.95)
    BR_PROPERTY(int, maxDepth, 1)
    BR_PROPERTY(int, maxWeakCount, 100)

    CascadeDataStorage *storage;
    CascadeBoostParams params;
    CascadeBoost boost;

    void train(const QList<Mat> &images, const QList<float> &labels)
    {
        foreach (float label, labels)
            if (!(label == 1.0f || label == 0.0f))
                qFatal("Labels for boosting must be 1 (POS) or 0 (NEG)");

        representation->train(images, labels);

        storage = new CascadeDataStorage(representation, images.size());
        fillStorage(images, labels);

        boost.clear(); // clear old data if necessary

        params = CascadeBoostParams(boostType, 0, minTAR, maxFAR, trimRate, maxDepth, maxWeakCount);
        if (!boost.train(storage, images.size(), precalcBufSize, precalcBufSize, params))
            qFatal("Unable to train Boosted Classifier");    

        // free the train data. Note storage keeps enough space for one image alloc'ed
        storage->freeTrainData();
        boost.freeTrainData();
    }

    float classify(const Mat &image) const
    {
        storage->setImage(image, -1, 0);
        return boost.predict(0, true);
    }

    void store(QDataStream &stream) const
    {
        params.store(stream);

        // Create local file
        QTemporaryFile tempFile;
        tempFile.open();
        tempFile.close();

        // Save MLP to local file
        boost.save(qPrintable(tempFile.fileName()));

        // Copy local file contents to stream
        tempFile.open();
        QByteArray data = tempFile.readAll();
        tempFile.close();
        stream << data;
    }

    void load(QDataStream &stream)
    {
        params.load(stream);
        storage = new CascadeDataStorage(representation, 1);

        // Copy local file contents from stream
        QByteArray data;
        stream >> data;

        // Create local file
        QTemporaryFile tempFile;
        tempFile.open();
        tempFile.write(data);
        tempFile.close();

        // Load MLP from local file
        boost.load(qPrintable(tempFile.fileName()), storage, params);
    }

    void finalize()
    {
        delete storage;
    }

private:
    void parallelFill(const Mat &image, float label, int idx)
    {
        storage->setImage(image, label, idx);
    }

    void fillStorage(const QList<Mat> &images, const QList<float> &labels)
    {
        //QFutureSynchronizer<void> sync;
        for (int i = 0; i < images.size(); i++) {
            //sync.addFuture(QtConcurrent::run(this, &BoostClassifier::parallelFill, images[i], labels[i], i));
            storage->setImage(images[i], labels[i], i);
            printf("Filled: %d / %d\r", i+1, images.size()); fflush(stdout);
        }
    }
};

BR_REGISTER(Classifier, BoostClassifier)

}

#include "boost.moc"
