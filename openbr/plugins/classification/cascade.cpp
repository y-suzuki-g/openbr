#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;
using namespace std;

namespace br
{

class CascadeClassifierTransform : public Transform
{
    Q_OBJECT

    Q_PROPERTY(QString description READ get_description WRITE set_description RESET reset_description STORED false)
    Q_PROPERTY(int numStages READ get_numStages WRITE set_numStages RESET reset_numStages STORED false)
    Q_PROPERTY(float maxFAR READ get_maxFAR WRITE set_maxFAR RESET reset_maxFAR STORED false)
    Q_PROPERTY(bool returnConfidence READ get_returnConfidence WRITE set_returnConfidence RESET reset_returnConfidence STORED false)
    Q_PROPERTY(bool ROCMode READ get_ROCMode WRITE set_ROCMode RESET reset_ROCMode STORED false)
    BR_PROPERTY(QString, description, "")
    BR_PROPERTY(int, numStages, 20)
    BR_PROPERTY(float, maxFAR, 0.005)
    BR_PROPERTY(bool, returnConfidence, true)
    BR_PROPERTY(bool, ROCMode, false)

    QList<Classifier*> stages;

    void train(const TemplateList &data)
    {
        if (description.isEmpty())
            qFatal("Need a classifier description!");

        QList<Mat> images;
        foreach (const Template &t, data)
            images.append(t);
        QList<float> labels = File::get<float>(data, "Label");

        const int numPos = labels.count(1.0f);
        const int numNeg = (int)labels.size() - numPos;

        QDateTime start = QDateTime::currentDateTime();

        for (int i = 0; i < numStages; i++) {
            qDebug() << "\n===== TRAINING STAGE" << i+1 << "=====";
            qDebug() << "<BEGIN";

            printStats(labels, numPos, numNeg);

            Classifier *tempStage = Factory<Classifier>::make("." + description);
            tempStage->train(images, labels);
            stages.append(tempStage);

            if (!updateTrainingSet(images, labels, numNeg))
                break;

            qDebug() << "END>";

            // Output training time up till now
            QDateTime now = QDateTime::currentDateTime();
            int seconds = start.secsTo(now);
            int days = seconds / 60 / 60 / 24;
            int hours = (seconds / 60 / 60) % 24;
            int minutes = (seconds / 60) % 60;
            int seconds_left = seconds % 60;
            qDebug("Training until now has taken %d days %d hours %d minutes and %d seconds", days, hours, minutes, seconds_left);
        }

        if (stages.size() == 0)
            qFatal("Training failed. Check training data");
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        float confidence = classify(src);

        if (returnConfidence) {
            dst.m() = Mat(1, 1, CV_32F);
            dst.m().at<float>(0,0) = confidence;
        } else {
            dst.file.set("Confidence", confidence);
        }
    }

    void store(QDataStream &stream) const
    {
        stream << stages.size();
        foreach (const Classifier *stage, stages)
            stage->store(stream);
    }

    void load(QDataStream &stream)
    {
        int _numStages;
        stream >> _numStages;
        for (int i = 0; i < _numStages; i++) {
            Classifier *tempStage = Factory<Classifier>::make("." + description);
            tempStage->load(stream);
            stages.append(tempStage);
        }
    }

private:
    float classify(const Mat &image) const
    {
        int i; float val;
        for (i = 0; i < stages.size(); i++)
            if ((val = stages[i]->classify(image)) < 0.0f)
                break;

        if (!ROCMode && i < stages.size())
            return -1.;
        return val * i;
    }

    void printStats(QList<float> &labels, int numPos, int numNeg)
    {
        float TAR = (float)labels.count(1.0f) / (float)numPos;
        float FAR = (float)labels.count(0.0f) / (float)numNeg;

        cout << "+----+---------+---------+" << endl;
        cout << "|  L |  Using  |    %    |" << endl;
        cout << "+----+---------+---------+" << endl;
        cout << "| POS|"; cout.width(9); cout << right << labels.count(1.0f) << "|"; cout.width(9); cout << right << TAR << "|" << endl;
        cout << "| NEG|"; cout.width(9); cout << right << labels.count(0.0f) << "|"; cout.width(9); cout << right << FAR << "|" << endl;
        cout << "+----+---------+---------+" << endl;
    }

    bool updateTrainingSet(QList<Mat> &images, QList<float> &labels, int numNeg)
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
            qDebug() << "END>";
            return false;
        }

        float FAR = ( (float)passedNeg / (float)numNeg );

        if (FAR < maxFAR) {
            qDebug("FAR is belowed desired level. Training is finished!");
            qDebug() << "END>";
            return false;
        }
        return true;
    }
};

BR_REGISTER(Transform, CascadeClassifierTransform)

} // namespace br

#include "classification/cascade.moc"
