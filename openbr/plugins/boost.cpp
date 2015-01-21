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
    Q_PROPERTY(float minTAR READ get_minTAR WRITE set_minTAR RESET reset_minTAR STORED false)
    Q_PROPERTY(float maxFAR READ get_maxFAR WRITE set_maxFAR RESET reset_maxFAR STORED false)
    Q_PROPERTY(float trimRate READ get_trimRate WRITE set_trimRate RESET reset_trimRate STORED false)
    Q_PROPERTY(int maxDepth READ get_maxDepth WRITE set_maxDepth RESET reset_maxDepth STORED false)
    Q_PROPERTY(int maxWeakCount READ get_maxWeakCount WRITE set_maxWeakCount RESET reset_maxWeakCount STORED false)
    BR_PROPERTY(br::Representation*, representation, NULL)
    BR_PROPERTY(Type, boostType, Gentle)
    BR_PROPERTY(float, minTAR, 0.995)
    BR_PROPERTY(float, maxFAR, 0.5)
    BR_PROPERTY(float, trimRate, 0.95)
    BR_PROPERTY(int, maxDepth, 1)
    BR_PROPERTY(int, maxWeakCount, 100)

    CascadeBoost boost;

    void train(const QList<Mat> &images, const QList<float> &labels)
    {
        CascadeDataStorage *storage = new CascadeDataStorage(representation->numFeatures(), images.size());
        fillStorage(storage, images, labels);

        CascadeBoostParams params(boostType, 0, minTAR, maxFAR, trimRate, maxDepth, maxWeakCount);
        if (!boost.train(storage, images.size(), params))
            qFatal("Unable to train Boosted Classifier");
    }

    float classify(const Mat &image, bool returnSum) const
    {

    }

private:
    void parallelFill(CascadeDataStorage *storage, const Mat &image, float label, int idx)
    {
        Mat sample = representation->evaluate(image);
        storage->setImage(sample, label, idx);
    }

    void fillStorage(CascadeDataStorage *storage, const QList<Mat> &images, const QList<float> &labels)
    {
        QFutureSynchronizer<void> sync;
        for (int i = 0; i < images.size(); i++) {
            sync.addFuture(QtConcurrent::run(this, &BoostClassifier::parallelFill, storage, images[i], labels[i], i));
            printf("Filled: %d / %d\r", i, images.size());
        }
    }
};

}

#include "boost.moc"
