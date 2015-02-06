#include "openbr_internal.h"
#include <openbr/openbr_plugin.h>
#include <openbr/core/boost.h>
#include <openbr/core/opencvutils.h>
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
    BR_PROPERTY(int, precalcBufSize, 512)
    BR_PROPERTY(float, minTAR, 0.995)
    BR_PROPERTY(float, maxFAR, 0.5)
    BR_PROPERTY(float, trimRate, 0.95)
    BR_PROPERTY(int, maxDepth, 1)
    BR_PROPERTY(int, maxWeakCount, 100)

    CascadeBoost boost;

    void train(const QList<Mat> &images, const QList<float> &labels)
    {
        foreach (float label, labels)
            if (!(label == 1.0f || label == 0.0f))
                qFatal("Labels for boosting must be 1 (POS) or 0 (NEG)");

        representation->train(images, labels);

        boost.clear(); // clear old data if necessary

        Mat data = images2FV(images);
        Mat _labels = OpenCVUtils::toMat(labels, 1);

        CascadeBoostParams params(boostType, maxWeakCount, trimRate, maxDepth, minTAR, maxFAR);
        if (!boost.train( data, _labels, params, representation ))
            qFatal("Unable to train Boosted Classifier");    
    }

    float classify(const Mat &image) const
    {
        return boost.predict(representation->preprocess(image), true);
    }

    void store(QDataStream &stream) const
    {
        boost.store(stream);
    }

    void load(QDataStream &stream)
    {
        boost.load( representation, stream );
    }

    void finalize()
    {
        boost.freeTrees();
    }

private:
    void parallelEnroll(Mat &data, const Mat &sample, int idx)
    {
        Mat pp = representation->preprocess(sample);
        Mat featureVector = representation->evaluate(pp);
        featureVector.copyTo(data.row(idx));
    }

    Mat images2FV(const QList<Mat> &images)
    {
        qDebug() << "Converting images to feature vectors...";
        Mat data(images.size(), representation->numFeatures(), CV_32F);
        QFutureSynchronizer<void> futures;
        for (int i = 0; i < images.size(); i++)
            futures.addFuture(QtConcurrent::run(this, &BoostClassifier::parallelEnroll, data, images[i], i));
        futures.waitForFinished();
        qDebug() << "All images have been converted";
        return data;
    }
};

BR_REGISTER(Classifier, BoostClassifier)

}

#include "boost.moc"
