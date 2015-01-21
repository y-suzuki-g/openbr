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

    Q_PROPERTY(QString description READ get_description WRITE set_description RESET reset_description STORED false)
    Q_PROPERTY(int numStages READ get_numStages WRITE set_numStages RESET reset_numStages STORED false)
    Q_PROPERTY(float maxFAR READ get_maxFAR WRITE set_maxFAR RESET reset_maxFAR STORED false)
    BR_PROPERTY(QString, description, "")
    BR_PROPERTY(int, numStages, 20)
    BR_PROPERTY(float, maxFAR, 0.005)

    QList<Classifier*> stages;

    void train(const QList<cv::Mat> &images, const QList<float> &labels)
    {
        int numPos = labels.count(1.0f);
        int numNeg = (int)labels.size() - numPos;
        double currFar;

        const clock_t begin_time = clock();

        for (int i = 0; i < numStages; i++) {
            qDebug() << "\n===== TRAINING" << i << "stage =====";
            qDebug() << "<BEGIN";
            if (!updateTrainingSet(currFar, images, labels, numPos, numNeg)) {
                qDebug() << "Train dataset for temp stage can not be filled. Branch training terminated.";
                break;
            }
            if (currFAR <= maxFAR) {
                qDebug() << "Required leaf false alarm rate achieved. Branch training terminated.";
                break;
            }

            Classifier *tempStage = Factory<Classifier>::make(description);
            tempStage->train(images, labels);

            qDebug() << "END>";

            stages.append(tempStage);

            // Output training time up till now
            float seconds = float( clock() - begin_time ) / CLOCKS_PER_SEC;
            int days = int(seconds) / 60 / 60 / 24;
            int hours = (int(seconds) / 60 / 60) % 24;
            int minutes = (int(seconds) / 60) % 60;
            int seconds_left = int(seconds) % 60;
            qDebug() << "Training until now has taken " << days << " days " << hours << " hours " << minutes << " minutes " << seconds_left <<" seconds." << endl;
        }

        if (stages.size() == 0)
            qFatal("Training failed. Check training data");
    }

    float classify(const Mat &image, bool returnSum) const
    {
        for (int i = 0; i < stages.size() - 1; i++)
            if (stage->classify(image) == 0.0f)
                return 0.0f;
        return stages.last()->classify(image, returnSum);
    }

private:
    bool updateTrainingSet(double& currFAR, QList<Mat> &images, QList<float> &labels, int numPos, int numNeg)
    {
        int passedPos = 0, passedNeg = 0;
        for (int i = 0; i < images.size(); i++) {
            if (predictPrecalc(images[i]) == 0.0f) {
                images.removeAt(i);
                labels.removeAt(i);
            } else {
                labels[i] == 0.0f ? passedNeg++ : passedPos++;
            }
        }

        if (passedPos == 0 || passedNeg == 0)
            return false;

        currFAR = ( (double)passedNeg / (double)numNeg );
        qDebug() << "POS passed : total    " << passedPos << ":" << numPos;
        qDebug() << "NEG passed : FAR    " << passedNeg << ":" << currFAR;

        return true;
    }

    float predictPrecalc(const Mat &image)
    {
        foreach (const Classifier *stage, stages)
            if (stage->classify(image) == 0.0f)
                return 0.0f;
        return 1.0f;
    }
};

BR_REGISTER(Transform, CascadeTransform)

} // namespace br

#include "cascade.moc"
