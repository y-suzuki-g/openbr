#include "openbr_internal.h"
#include "openbr/openbr_plugin.h"
#include "openbr/core/boost.h"
#include "openbr/core/opencvutils.h"

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

    void train(const QList<Mat> &images, const QList<float> &labels)
    {
        if (description.isEmpty())
            qFatal("Need a classifier description!");

        QList<Mat> _images(images); // mutable copy
        QList<float> _labels(labels); // mutable copy

        int numPos = labels.count(1.0f);
        int numNeg = (int)labels.size() - numPos;

        const clock_t begin_time = clock();

        for (int i = 0; i < numStages; i++) {
            qDebug() << "\n===== TRAINING" << i << "stage =====";
            qDebug() << "<BEGIN";

            if (!updateTrainingSet(_images, _labels, numPos, numNeg))
                break;

            Classifier *tempStage = Factory<Classifier>::make("." + description);
            tempStage->train(_images, _labels);

            qDebug() << "END>";

            stages.append(tempStage);

            // Output training time up till now
            float seconds = float( clock() - begin_time ) / CLOCKS_PER_SEC;
            int days = int(seconds) / 60 / 60 / 24;
            int hours = (int(seconds) / 60 / 60) % 24;
            int minutes = (int(seconds) / 60) % 60;
            int seconds_left = int(seconds) % 60;
            qDebug("Training until now has taken %d days %d hours %d minutes and %d seconds", days, hours, minutes, seconds_left);
        }

        if (stages.size() == 0)
            qFatal("Training failed. Check training data");
    }

    float classify(const Mat &image) const
    {
        float val = 0.;
        for (int i = 0; i < stages.size(); i++)
            if ((val = stages[i]->classify(image)) < 0.0f)
                return val;
        return val;
    }

private:
    bool updateTrainingSet(QList<Mat> &images, QList<float> &labels, int numPos, int numNeg)
    {
        int passedPos = 0, passedNeg = 0, i = 0;
        while (i < images.size()) {
            if (classify(images[i]) < 0.0f) {
                images.removeAt(i);
                labels.removeAt(i);
            } else {
                labels[i] == 0.0f ? passedNeg++ : passedPos++;
                i++;
            }
        }

        if (passedPos == 0 || passedNeg == 0) {
            qDebug("Unable to update training set. Remaining samples: %d POS and %d NEG", passedPos, passedNeg);
            return false;
        }

        float FAR = ( (float)passedNeg / (float)numNeg );

        if (FAR < maxFAR) {
            qDebug("FAR is belowed desired level. Training is finished!");
            return false;
        }

        cout << "+----+---------+---------+" << endl;
        cout << "|  L |  Passed |    %    |" << endl;
        cout << "+----+---------+---------+" << endl;
        cout << "| POS|"; cout.width(9); cout << right << passedPos << "|"; cout.width(9); cout << right << (double)passedPos / (double)numPos << "|" << endl;
        cout << "| NEG|"; cout.width(9); cout << right << passedNeg << "|"; cout.width(9); cout << right << FAR << "|" << endl;
        cout << "+----+---------+---------+" << endl;

        return true;
    }
};

BR_REGISTER(Classifier, CascadeClassifier)

class CascadeTest : public Transform
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

BR_REGISTER(Transform, CascadeTest)

} // namespace br

#include "cascade.moc"
