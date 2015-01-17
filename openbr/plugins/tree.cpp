#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core_c.h>

#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"

#include <QString>
#include <QTemporaryFile>

using namespace std;
using namespace cv;

namespace br
{

static void storeModel(const CvStatModel &model, QDataStream &stream)
{
    // Create local file
    QTemporaryFile tempFile;
    tempFile.open();
    tempFile.close();

    // Save MLP to local file
    model.save(qPrintable(tempFile.fileName()));

    // Copy local file contents to stream
    tempFile.open();
    QByteArray data = tempFile.readAll();
    tempFile.close();
    stream << data;
}

static void loadModel(CvStatModel &model, QDataStream &stream)
{
    // Copy local file contents from stream
    QByteArray data;
    stream >> data;

    // Create local file
    QTemporaryFile tempFile(QDir::tempPath()+"/model");
    tempFile.open();
    tempFile.write(data);
    tempFile.close();

    // Load MLP from local file
    model.load(qPrintable(tempFile.fileName()));
}

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV's random trees framework
 * \author Scott Klum \cite sklum
 * \brief http://docs.opencv.org/modules/ml/doc/random_trees.html
 */
class ForestTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(bool classification READ get_classification WRITE set_classification RESET reset_classification STORED false)
    Q_PROPERTY(float splitPercentage READ get_splitPercentage WRITE set_splitPercentage RESET reset_splitPercentage STORED false)
    Q_PROPERTY(int maxDepth READ get_maxDepth WRITE set_maxDepth RESET reset_maxDepth STORED false)
    Q_PROPERTY(int maxTrees READ get_maxTrees WRITE set_maxTrees RESET reset_maxTrees STORED false)
    Q_PROPERTY(float forestAccuracy READ get_forestAccuracy WRITE set_forestAccuracy RESET reset_forestAccuracy STORED false)
    Q_PROPERTY(bool returnConfidence READ get_returnConfidence WRITE set_returnConfidence RESET reset_returnConfidence STORED false)
    Q_PROPERTY(bool overwriteMat READ get_overwriteMat WRITE set_overwriteMat RESET reset_overwriteMat STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    Q_PROPERTY(QString outputVariable READ get_outputVariable WRITE set_outputVariable RESET reset_outputVariable STORED false)
    BR_PROPERTY(bool, classification, true)
    BR_PROPERTY(float, splitPercentage, .01)
    BR_PROPERTY(int, maxDepth, std::numeric_limits<int>::max())
    BR_PROPERTY(int, maxTrees, 10)
    BR_PROPERTY(float, forestAccuracy, .1)
    BR_PROPERTY(bool, returnConfidence, true)
    BR_PROPERTY(bool, overwriteMat, true)
    BR_PROPERTY(QString, inputVariable, "Label")
    BR_PROPERTY(QString, outputVariable, "")

    CvRTrees forest;

    void train(const TemplateList &data)
    {
        Mat samples = OpenCVUtils::toMat(data.data());
        Mat labels = OpenCVUtils::toMat(File::get<float>(data, inputVariable));

        Mat types = Mat(samples.cols + 1, 1, CV_8U);
        types.setTo(Scalar(CV_VAR_NUMERICAL));

        if (classification) {
            types.at<char>(samples.cols, 0) = CV_VAR_CATEGORICAL;
        } else {
            types.at<char>(samples.cols, 0) = CV_VAR_NUMERICAL;
        }

        int minSamplesForSplit = data.size()*splitPercentage;
        forest.train( samples, CV_ROW_SAMPLE, labels, Mat(), Mat(), types, Mat(),
                    CvRTParams(maxDepth,
                               minSamplesForSplit,
                               0,
                               false,
                               2,
                               0, // priors
                               false,
                               0,
                               maxTrees,
                               forestAccuracy,
                               CV_TERMCRIT_ITER | CV_TERMCRIT_EPS));

        qDebug() << "Number of trees:" << forest.get_tree_count();
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        float response;
        if (classification && returnConfidence) {
            // Fuzzy class label
            response = forest.predict_prob(src.m().reshape(1,1));
        } else {
            response = forest.predict(src.m().reshape(1,1));
        }

        if (overwriteMat) {
            dst.m() = Mat(1, 1, CV_32F);
            dst.m().at<float>(0, 0) = response;
        } else {
            dst.file.set(outputVariable, response);
        }
    }

    void load(QDataStream &stream)
    {
        loadModel(forest,stream);
    }

    void store(QDataStream &stream) const
    {
        storeModel(forest,stream);
    }

    void init()
    {
        if (outputVariable.isEmpty())
            outputVariable = inputVariable;
    }
};

BR_REGISTER(Transform, ForestTransform)
/*
class Stage
{
public:
    Stage() { threshold = -FLT_MAX; params = CvBoostParams(); }
    Stage(const CvBoostParams &_params) { threshold = -FLT_MAX; params = _params; }

    ~Stage() {}

    QList<int> feature_indices() const { return featureIndicesList; }

    bool train(const Mat &data, const Mat &labels, const Mat &samp_mask, float minTAR)
    {
        featureIndicesMask = Mat::zeros(1, data.cols, data.type());
        if (!boost.train(data, CV_ROW_SAMPLE, labels, Mat(), samp_mask, Mat(), Mat(), params, true))
            return false;

        updateVisited();
        calcThreshold(data, labels, minTAR);

        return true;
    }

    float predict(const Mat &response, bool useMask = false, bool raw = false) const
    {
        float val = sum(useMask ? mask(response) : response)[0];
        if (raw)
            return val;
        return val - threshold > FLT_EPSILON ? 1.0f : -1.0f;
    }

    void load(QDataStream &stream)
    {
        loadModel(boost, stream);
        params = boost.get_params();
        stream >> threshold;
        stream >> featureIndicesList;
        stream >> featureIndicesMask;
    }

    void store(QDataStream &stream) const
    {
        storeModel(boost, stream);
        stream << threshold;
        stream << featureIndicesList;
        stream << featureIndicesMask;
    }

private:
    CvBoost boost;
    CvBoostParams params;
    float threshold;
    QList<int> featureIndicesList;
    Mat featureIndicesMask;

    void updateVisited()
    {
        CvSeq *classifiers = boost.get_weak_predictors();
        CvSeqReader reader;
        cvStartReadSeq(classifiers, &reader);
        cvSetSeqReaderPos(&reader, classifiers->total - 1);
        for (int i = 0; i < classifiers->total - 1; i++) {
            CvBoostTree* tree;
            CV_READ_SEQ_ELEM(tree, reader);
            int visitedIdx = tree->get_root()->split->var_idx;
            featureIndicesList.append(visitedIdx); // local feature list
            featureIndicesMask.at<float>(visitedIdx) = 1.0f; // local feature map float to match data type
        }
    }

    inline Mat mask(const Mat &input) const
    {
        Mat response;
        multiply(input, featureIndicesMask, response);
        return response;
    }

    void calcThreshold(const Mat &data, const Mat &labels, float minTAR)
    {
        QList<float> preds;
        for (int i = 0; i < data.rows; i++)
            if (labels.at<int>(i) == 1)
                preds.append(predict(data.row(i), true, true));

        sort(preds.begin(), preds.end());
        int threshIdx = (int)((1.0f - minTAR) * preds.size());
        threshold = preds[threshIdx];
    }
};
*/
/*!
 * \ingroup transforms
 * \brief Wraps OpenCV's Ada Boost framework
 * \author Scott Klum \cite sklum
 * \brief http://docs.opencv.org/modules/ml/doc/boosting.html
 */
 /*
class CascadeClassifier : public Classifier
{
    Q_OBJECT
    Q_ENUMS(Type)
    Q_ENUMS(SplitCriteria)

    Q_PROPERTY(br::Representation* rep READ get_rep WRITE set_rep RESET reset_rep STORED false)
    Q_PROPERTY(bool ROCMode READ get_ROCMode WRITE set_ROCMode RESET reset_ROCMode STORED false)
    Q_PROPERTY(int numStages READ get_numStages WRITE set_numStages RESET reset_numStages STORED false)
    Q_PROPERTY(float minTAR READ get_minTAR WRITE set_minTAR RESET reset_minTAR STORED false)
    Q_PROPERTY(float maxFAR READ get_maxFAR WRITE set_maxFAR RESET reset_maxFAR STORED false)
    Q_PROPERTY(Type type READ get_type WRITE set_type RESET reset_type STORED false)
    Q_PROPERTY(SplitCriteria splitCriteria READ get_splitCriteria WRITE set_splitCriteria RESET reset_splitCriteria STORED false)
    Q_PROPERTY(int weakCount READ get_weakCount WRITE set_weakCount RESET reset_weakCount STORED false)
    Q_PROPERTY(float trimRate READ get_trimRate WRITE set_trimRate RESET reset_trimRate STORED false)
    Q_PROPERTY(int folds READ get_folds WRITE set_folds RESET reset_folds STORED false)
    Q_PROPERTY(int maxDepth READ get_maxDepth WRITE set_maxDepth RESET reset_maxDepth STORED false)

public:
    enum Type { Discrete = CvBoost::DISCRETE,
                Real = CvBoost::REAL,
                Logit = CvBoost::LOGIT,
                Gentle = CvBoost::GENTLE};

    enum SplitCriteria { Default = CvBoost::DEFAULT,
                         Gini = CvBoost::GINI,
                         Misclass = CvBoost::MISCLASS,
                         Sqerr = CvBoost::SQERR};

private:
    BR_PROPERTY(br::Representation*, rep, NULL)
    BR_PROPERTY(bool, ROCMode, false)
    BR_PROPERTY(int, numStages, 20)
    BR_PROPERTY(float, minTAR, 0.995f)
    BR_PROPERTY(float, maxFAR, 0.5f)
    BR_PROPERTY(Type, type, Gentle)
    BR_PROPERTY(SplitCriteria, splitCriteria, Default)
    BR_PROPERTY(int, weakCount, 100)
    BR_PROPERTY(float, trimRate, .95)
    BR_PROPERTY(int, folds, 0)
    BR_PROPERTY(int, maxDepth, 1)

    QList<Stage *> stages;

    void finalize()
    {
        for (int i = 0; i < stages.size(); i++)
            delete stages[i];
    }

    void train(const QList<Mat> &images, const QList<float> &labels)
    {
        Mat data(images.size(), rep->numFeatures(), CV_32FC1);
        Mat _labels = OpenCVUtils::toMat(labels, 1);
        _labels.convertTo(_labels, CV_32SC1);

        for (int i = 0; i < images.size(); i++) {
            Mat image = rep->preprocess(images[i]);
            rep->evaluate(image).copyTo(data.row(i));
        }

        CvBoostParams params;
        params.boost_type = type;
        params.split_criteria = splitCriteria;
        params.weak_count = weakCount;
        params.weight_trim_rate = trimRate;
        params.cv_folds = folds;
        params.max_depth = maxDepth;

        Mat samp_mask(data.rows, 1, CV_8UC1);

        for (int ns = 0; ns < numStages; ns++) {
            qDebug("\n");
            qDebug("+-----+------+---------------+---------------+");
            qDebug("|  S  |  wc  |      TAR      |      FAR      |");
            qDebug("+-----+------+---------------+---------------+");

            Stage *stage = new Stage(params);
            if (!stage->train(data, _labels, samp_mask, minTAR))
                return;
            stages.append(stage);
            calcErr(data, _labels);
            updateTrainData(data, samp_mask);
        }
    }

    float classify(const Mat &image) const
    {
        Mat img = rep->preprocess(image);
        foreach (const Stage *stage, stages) {
            Mat response = rep->evaluate(img, stage->feature_indices());
            if (stage->predict(response) == -1.0f)
                return -1.0f;
        }
        return 1.0f;
    }

    void load(QDataStream &stream)
    {
        int numStages;
        stream >> numStages;
        for (int i = 0; i < numStages; i++) {
            Stage *stage; stage->load(stream);
            stages.append(stage);
        }
    }

    void store(QDataStream &stream) const
    {
        stream << stages.size();
        foreach (const Stage *stage, stages)
            stage->store(stream);
    }

    void calcErr(const Mat &data, const Mat &labels) const
    {
        float TAR, FAR;
        int posWrong = 0, totalPos = 0;
        int negRight = 0, totalNeg = 0;

        for (int i = 0; i < data.rows; i++) {
            int label = labels.at<int>(i);
            label == 1 ? totalPos++ : totalNeg++;
            for (int j = 0; j < stages.size(); j++) {
                if (stages[j]->predict(data.row(i), true) != 1.0f) {
                    if (labels.at<int>(i) == 1)
                        posWrong++;
                    else
                        negRight++;
                }
            }
        }

        TAR = 1 - ((float)posWrong / totalPos);
        FAR = 1 - ((float)negRight / totalNeg);

        int totalWeakCount = 0;
        foreach (const Stage *s, stages)
            totalWeakCount += s->feature_indices().size();

        qDebug("| %.2d | %.4d | %.11f | %.11f |", stages.size(), totalWeakCount, TAR, FAR);
        qDebug("+-----+------+---------------+---------------+");
    }

    void updateTrainData(const Mat &data, Mat &samp_mask)
    {
        Stage *stage = stages.last();
        for (int i = 0; i < samp_mask.rows; i++)
            if ((samp_mask.at<uchar>(i) != 0) && (stage->predict(data.row(i), true, false) == -1.0f))
                samp_mask.at<uchar>(i) = (uchar)0;
    }
};

//BR_REGISTER(Classifier, CascadeClassifier)

class CascadeTestTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Classifier* classifier READ get_classifier WRITE set_classifier RESET reset_classifier STORED false)
    BR_PROPERTY(br::Classifier*, classifier, NULL)

    void train(const TemplateList &data)
    {
        QList<Mat> images;
        foreach (const Template &t, data)
            images.append(t.m());

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
*/
} // namespace br

#include "tree.moc"
