#include "openbr_internal.h"
#include "openbr/openbr_plugin.h"
#include "openbr/core/boost.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/qtutils.h"

#include <QtConcurrent>

using namespace cv;
using namespace std;

namespace br
{

class CascadeClassifier : public Classifier
{
    Q_OBJECT

public:
    Q_ENUMS(Type)

    // Cascade params
    Q_PROPERTY(br::Representation* representation READ get_representation WRITE set_representation RESET reset_representation STORED false)
    Q_PROPERTY(int winWidth READ get_winWidth WRITE set_winWidth RESET reset_winWidth STORED false)
    Q_PROPERTY(int winHeight READ get_winHeight WRITE set_winHeight RESET reset_winHeight STORED false)
    Q_PROPERTY(int numStages READ get_numStages WRITE set_numStages RESET reset_numStages STORED false)
    BR_PROPERTY(br::Representation*, representation, NULL)
    BR_PROPERTY(int, winWidth, 24)
    BR_PROPERTY(int, winHeight, 24)
    BR_PROPERTY(int, numStages, 20)

    enum Type { Discrete = CvBoost::DISCRETE,
                Real = CvBoost::REAL,
                Logit = CvBoost::LOGIT,
                Gentle = CvBoost::GENTLE };

    // Stage params
    Q_PROPERTY(Type boostType READ get_boostType WRITE set_boostType RESET reset_boostType STORED false)
    Q_PROPERTY(float minTAR READ get_minTAR WRITE set_minTAR RESET reset_minTAR STORED false)
    Q_PROPERTY(float maxFAR READ get_maxFAR WRITE set_maxFAR RESET reset_maxFAR STORED false)
    Q_PROPERTY(float trimRate READ get_trimRate WRITE set_trimRate RESET reset_trimRate STORED false)
    Q_PROPERTY(int maxDepth READ get_maxDepth WRITE set_maxDepth RESET reset_maxDepth STORED false)
    Q_PROPERTY(int maxWeakCount READ get_maxWeakCount WRITE set_maxWeakCount RESET reset_maxWeakCount STORED false)
    BR_PROPERTY(Type, boostType, Gentle)
    BR_PROPERTY(float, minTAR, 0.995)
    BR_PROPERTY(float, maxFAR, 0.5)
    BR_PROPERTY(float, trimRate, 0.95)
    BR_PROPERTY(int, maxDepth, 1)
    BR_PROPERTY(int, maxWeakCount, 100)

    // training values
    QList<CascadeBoost*> stages;
    CascadeDataStorage *storage;
    int numNeg, numPos;

    void train(const QList<Mat> &images, const QList<float> &labels)
    {
        const clock_t begin_time = clock();

        foreach (float label, labels)
            if (label != 1.0f && label != 0.0f)
                qFatal("Labels must be either 1 (POS) or 0 (NEG) for boosting");

        numPos = labels.count(1.0f);
        numNeg = (int)labels.size() - numPos;
        int numSamples = numPos + numNeg;

        storage = new CascadeDataStorage(representation->numFeatures(), images.size());
        fillStorage(images, labels);

        double requiredLeafFARate = pow((double)maxFAR, (double)numStages) / (double)maxDepth;
        double tempLeafFARate;

        CascadeBoostParams params(boostType, 0, minTAR, maxFAR, trimRate, maxDepth, maxWeakCount);

        for (int i = 0; i < numStages; i++) {
            qDebug() << "\n===== TRAINING" << i << "stage =====";
            qDebug() << "<BEGIN";
            if (!updateTrainingSet(tempLeafFARate, numSamples)) {
                qDebug() << "Train dataset for temp stage can not be filled. Branch training terminated.";
                break;
            }
            if (tempLeafFARate <= requiredLeafFARate) {
                qDebug() << "Required leaf false alarm rate achieved. Branch training terminated.";
                break;
            }

            CascadeBoost* tempStage = new CascadeBoost;
            bool isStageTrained = tempStage->train( storage, numSamples, params);

            qDebug() << "END>";

            if(!isStageTrained)
                break;

            stages.append(tempStage);

            // Output training time up till now
            float seconds = float( clock() - begin_time ) / CLOCKS_PER_SEC;
            int days = int(seconds) / 60 / 60 / 24;
            int hours = (int(seconds) / 60 / 60) % 24;
            int minutes = (int(seconds) / 60) % 60;
            int seconds_left = int(seconds) % 60;
            qDebug() << "Training until now has taken " << days << " days " << hours << " hours " << minutes << " minutes " << seconds_left <<" seconds." << endl;
        }

        delete storage;

        if(stages.size() == 0)
            qFatal("Cascade classifier can't be trained. Check the used training parameters.");
    }

    float classify(const Mat &image) const
    {
        (void)image;
        return 0.0f;
    }

    void store(QDataStream &stream) const
    {
        stream << numStages;
        stream << winWidth;
        stream << winHeight;
        foreach (const CascadeBoost *stage, stages)
            stage->store(stream);
    }

private:
    void parallelFill(const Mat &image, float label, int idx)
    {
        Mat sample = representation->evaluate(image);
        storage->setImage(sample, label, idx);
    }

    void fillStorage(const QList<Mat> &images, const QList<float> &labels)
    {
        qDebug() << "\nCreating Training Data...";

        numPos = labels.count(1.0f);
        numNeg = labels.size() - numPos;

        // TODO: should multithread this
        for (int i = 0; i < images.size(); i++) {
            QtConcurrent::run(this, &CascadeClassifier::parallelFill, images[i], labels[i], i);
            printf("Filled %d / %d\r", i+1, images.size());
        }
    }

    bool updateTrainingSet(double& acceptanceRatio, int &numSamples)
    {
        int passedPos = 0, passedNeg = 0, maxNumSamples = numSamples;
        bool updatedSampleCount = false;
        int idx = 0;
        for (int i = 0; i < maxNumSamples; idx++, i++) {
            while (idx < maxNumSamples) {
                float label = storage->getLabel(idx);
                if (label != 1.0f && !updatedSampleCount) { // don't keep to many neg images
                    updatedSampleCount = true;
                    maxNumSamples = passedPos + cvRound(((double)numNeg * (double)passedPos) / numPos);
                }

                if (predictPrecalc(idx) == 1.0f) {
                    label == 1.0f ? passedPos++ : passedNeg++;
                    storage->setImage(storage->data.row(idx), label, i);
                    break;
                }
                idx++;
            }
        }

        if (passedPos == 0 || passedNeg == 0)
            return false;

        numSamples = passedPos + passedNeg;
        acceptanceRatio = ( (double)passedNeg/(double)numNeg );
        qDebug() << "POS passed : total    " << passedPos << ":" << numPos;
        qDebug() << "NEG passed : acceptanceRatio    " << passedNeg << ":" << acceptanceRatio;

        return false;
    }

    int predictPrecalc(int sampleIdx)
    {
        foreach (const CascadeBoost *stage, stages) {
            if (stage->predict(sampleIdx) == 0.f)
                return 0;
        }
        return 1;
    }
};

BR_REGISTER(Classifier, CascadeClassifier)

class CascadeTestTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Classifier* classifier READ get_classifier WRITE set_classifier RESET reset_classifier STORED false)
    BR_PROPERTY(br::Classifier*, classifier, NULL)

    void train(const TemplateList &data)
    {
        QList<Mat> images;
        foreach (const Template &t, data)
            images.append(t);

        QList<float> labels = File::get<float>(data, "Label");

        classifier->train(images, labels);
    }

    void project(const Template &src, Template &dst) const
    {
        (void)src;
        (void)dst;
    }
};

BR_REGISTER(Transform, CascadeTestTransform)

} // namespace br

#include "cascade.moc"
