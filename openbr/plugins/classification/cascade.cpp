#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;
using namespace std;

namespace br
{

class CascadeClassifier : public Classifier
{
    Q_OBJECT

    Q_PROPERTY(br::Classifier *classifier READ get_classifier WRITE set_classifier RESET reset_classifier STORED false)
    Q_PROPERTY(int numNeg READ get_numNeg WRITE set_numNeg RESET reset_numNeg STORED false)
    Q_PROPERTY(int numStages READ get_numStages WRITE set_numStages RESET reset_numStages STORED false)
    Q_PROPERTY(float maxFAR READ get_maxFAR WRITE set_maxFAR RESET reset_maxFAR STORED false)
    Q_PROPERTY(bool ROCMode READ get_ROCMode WRITE set_ROCMode RESET reset_ROCMode STORED false)
    BR_PROPERTY(br::Classifier*, classifier, NULL)
    BR_PROPERTY(int, numNeg, 1000)
    BR_PROPERTY(int, numStages, 20)
    BR_PROPERTY(float, maxFAR, 0.005)
    BR_PROPERTY(bool, ROCMode, false)

    QList<Classifier*> stages;

    bool train(const QList<Mat> &_images, const QList<float> &_labels)
    {
        if (!classifier)
            qFatal("Need a classifier!");

        QList<Mat> posImages, negImages, trainingSet;
        QList<float> labels = _labels;

        for (int i = 0; i < _images.size(); i++)
            labels[i] == 1 ? posImages.append(_images[i]) : negImages.append(_images[i]);

        for (int i = 0; i < posImages.size(); i++) {
            Mat resizedPos;
            resize(posImages[i], resizedPos, windowSize());
            posImages.replace(i, resizedPos);
        }

        stages.clear();

        trainingSet.append(posImages);
        labels = QList<float>::fromVector(QVector<float>(posImages.size(), 1));
        for (int n = 0; n < numNeg; n++) {
            int idx = rand() % negImages.size();
            trainingSet.append(getRandomNeg(negImages[idx]));
            labels.append(0.);
        }

        QDateTime start = QDateTime::currentDateTime();

        int stageCounter = 0;
        do
        {
            qDebug() << "\n===== TRAINING STAGE" << stageCounter+1 << "=====";
            qDebug() << "<BEGIN";
            Classifier *tempStage = Factory<Classifier>::make("." + classifier->description(true));
            if (!tempStage->train(trainingSet, labels))
                break;
            stages.append(tempStage);

            qDebug() << "END>";

            // Output training time up till now
            QDateTime now = QDateTime::currentDateTime();
            int seconds = start.secsTo(now);
            int days = seconds / 60 / 60 / 24;
            int hours = (seconds / 60 / 60) % 24;
            int minutes = (seconds / 60) % 60;
            int seconds_left = seconds % 60;
            qDebug("Training until now has taken %d days %d hours %d minutes and %d seconds", days, hours, minutes, seconds_left);

            stageCounter++;
        } while (updateTrainingSet(posImages, negImages, trainingSet, labels) && (stageCounter < numStages));

        if (stages.size() == 0)
            return false;
        return true;
    }

    Mat preprocess(const Mat &image) const
    {
        return classifier->preprocess(image);
    }

    Size windowSize() const
    {
        return classifier->windowSize();
    }

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
        qDebug("numStages: %d", numStages);
        for (int i = 0; i < _numStages; i++) {
            Classifier *tempStage = Factory<Classifier>::make("." + classifier->description(true));
            tempStage->load(stream);
            stages.append(tempStage);
        }
    }

    void finalize()
    {

    }

private:
    bool updateTrainingSet(const QList<Mat> &posImages, const QList<Mat> &negImages, QList<Mat> &trainingSet, QList<float> &labels)
    {
        trainingSet.clear(); labels.clear();

        qDebug() << "Building Training Set...";

        // Collect passed pos
        foreach (const Mat &pos, posImages)
            if (classify(pos) > 0.0f) {
                trainingSet.append(pos);
                labels.append(1.);
            }

        qDebug("Pos => %d : %d (%f%%)", trainingSet.size(), posImages.size(), trainingSet.size() / (float)posImages.size());

        // find hard negatives
        QList<Mat> randomNegs = negImages;

        int checked = 0;
        for (int n = 0; n < numNeg; n++) {
            std::random_shuffle(randomNegs.begin(), randomNegs.end());
            Mat negSample; bool foundNegSample = false;
            foreach (const Mat &neg, randomNegs)
                if (getNegSample(negSample, neg, checked)) {
                    foundNegSample = true;
                    break;
                }

            if (!foundNegSample)
                break;

            trainingSet.append(negSample);
            labels.append(0.);

            printf("Neg => %d : %d\r", n+1, numNeg); std::fflush(stdout);
        }

        if (labels.count(1) == 0 || labels.count(0) < numNeg) {
            qDebug("Unable to update training set. Remaining samples: %d POS and %d NEG", labels.count(1), labels.count(0));
            qDebug() << "END>";
            return false;
        }

        float FAR = ( (float)numNeg / (float)checked );
        qDebug("\nAcceptance ratio: %f", FAR);

        if (FAR < maxFAR) {
            qDebug("FAR is belowed desired level. Training is finished!");
            qDebug() << "END>";
            return false;
        }
        return true;
    }

    Mat getRandomNeg(Mat &image)
    {
        Size winSize = windowSize();

        int x = rand() % (image.cols - winSize.width - 1);
        int y = rand() % (image.rows - winSize.height - 1);
        int w = qMax(rand() % (image.cols - x), winSize.width);
        int h = qMax(rand() % (image.rows - y), winSize.height);

        Mat sample = image(Rect(x, y, w, h)), resized;
        resize(sample, resized, winSize);
        return resized;
    }

    bool getNegSample(Mat &sample, const Mat &neg, int &checked) const
    {
        Size winSize = windowSize();

        int maxSize = qMax(neg.rows, neg.cols);

        const float scaleFrom = qMin(winSize.width/(float)maxSize, winSize.height/(float)maxSize);

        Mat scaledImage;
        for (float scale = scaleFrom; scale <= 1.; scale *= 1.2) {
            resize(neg, scaledImage, Size(), scale, scale);

            const int step = scale < 1. ? 1 : 2;
            for (int y = 0; y < (scaledImage.rows - winSize.height); y += step) {
                for (int x = 0; x < (scaledImage.cols - winSize.width); x += step) {
                    checked++;
                    if (classify(scaledImage(Rect(Point(x, y), winSize))) > 0.0f) {
                        Mat img = neg(Rect(qRound(x/scale),
                                           qRound(y/scale),
                                           qRound(winSize.width/scale),
                                           qRound(winSize.height/scale)));
                        resize(img, sample, winSize);
                        return true;
                    }
                }
            }
        }
        return false;
    }
};

BR_REGISTER(Classifier, CascadeClassifier)

} // namespace br

#include "classification/cascade.moc"
