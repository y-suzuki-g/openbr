#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/common.h"
#include "openbr/core/qtutils.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

using namespace cv;
using namespace Eigen;

namespace br
{

class SlidingWindowTransform : public MetaTransform
{
    Q_OBJECT

    Q_PROPERTY(br::Classifier *classifier READ get_classifier WRITE set_classifier RESET reset_classifier STORED false)
    Q_PROPERTY(int winWidth READ get_winWidth WRITE set_winWidth RESET reset_winWidth STORED false)
    Q_PROPERTY(int winHeight READ get_winHeight WRITE set_winHeight RESET reset_winHeight STORED false)
    Q_PROPERTY(int step READ get_step WRITE set_step RESET reset_step STORED false)
    Q_PROPERTY(int rounds READ get_rounds WRITE set_rounds RESET reset_rounds STORED false)
    Q_PROPERTY(int negsPerImage READ get_negsPerImage WRITE set_negsPerImage RESET reset_negsPerImage STORED false)
    BR_PROPERTY(br::Classifier*, classifier, NULL)
    BR_PROPERTY(int, winWidth, 24)
    BR_PROPERTY(int, winHeight, 24)
    BR_PROPERTY(int, step, 1)
    BR_PROPERTY(int, rounds, 1)
    BR_PROPERTY(int, negsPerImage, 5)

public:
    void train(const TemplateList &data)
    {
        QList<Mat> posImages, negImages;
        foreach (const Template &t, data)
            if (t.file.get<float>("Label") == 1.0f)
                posImages.append(t);
            else
                negImages.append(t);

        QList<Mat> negSamples = getRandomNegs(negImages);

        QList<float> posLabels = QList<float>::fromVector(QVector<float>(posImages.size(), 1.0f));
        QList<float> negLabels = QList<float>::fromVector(QVector<float>(negSamples.size(), 0.0f));

        QList<Mat> images = posImages; images.append(negSamples);
        QList<float> labels = posLabels; labels.append(negLabels);
        classifier->train(images, labels);

        // bootstrap
        for (int i = 1; i < rounds; i++) {
            qDebug() << "\n===== Bootstrapping Round" << i << " =====";
            negSamples.clear(); // maybe keep some neg samples from the previous round?
            foreach (const Mat &image, negImages) {
                int imageNegCount = 0;
                foreach (const Rect &rect, getRects(image)) {
                    if (imageNegCount >= negsPerImage)
                        break;
                    if (classifier->classify(image(rect)) >= 0.0f) {
                        negSamples.append(image(rect));
                        imageNegCount++;
                    }
                }
            }
            if (negSamples.size() == 0) {
                qDebug() << "No more negative samples can be pulled from the training data";
                break;
            }

            images = posImages; images.append(negSamples);
            labels = posLabels; labels.append(QList<float>::fromVector(QVector<float>(negSamples.size(), 0.0f)));
            classifier->train(images, labels);
        }
    }

    void project(const Template &src, Template &dst) const
    {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach (const Template &t, src) {
            QList<float> scales = t.file.getList<float>("Scales", QList<float>::fromVector(QVector<float>(t.size(), 1.0f)));

            QList<Rect> detections;
            QList<float> confidences;
            Mat orig;
            for (int i = 0; i < t.size(); i++) {
                Mat m = t[i];

                QList<Rect> rects = getRects(m);
                foreach (const Rect &rect, rects) {
                    float confidence = classifier->classify(m(rect));
                    if (confidence > 0.0f)
                        smartAddRect(detections, confidences, scaleRect(rect, scales[i]), confidence);
                }
                if (scales[i] == 1 && dst.size() == 0)
                    orig = m;
            }

            for (int i = 0; i < detections.size(); i++) {
                Template u(t.file, orig);
                u.file.set("Confidence", confidences[i]);
                u.file.appendRect(detections[i]);
                u.file.set("Rect", OpenCVUtils::fromRect(detections[i]));
                dst.append(u);
            }
        }
    }

    void load(QDataStream &stream)
    {
        classifier->load(stream);
    }

    void store(QDataStream &stream) const
    {
        classifier->store(stream);
    }

private:
    inline Rect scaleRect(const Rect &rect, float scale) const
    {
        return Rect(rect.x * scale, rect.y * scale, rect.width * scale, rect.height * scale);
    }

    // nonmaximum supression
    void smartAddRect(QList<Rect> &rects, QList<float> &confidences, const Rect &rect, float confidence) const
    {
        for (int i = 0; i < rects.size(); i++) {
            float overlap = OpenCVUtils::overlap(rects[i], rect);
            if (overlap > 0.5) {
                if (confidences[i] < confidence) {
                    rects.replace(i, rect);
                    confidences.replace(i, confidence);
                }
                return;
            }
        }
        rects.append(rect);
        confidences.append(confidence);
    }

    QList<Rect> getRects(const Mat &img) const
    {
        QList<Rect> rects;
        for (int x = 0; x < (img.cols - winWidth); x += step)
            for (int y = 0; y < (img.rows - winHeight); y += step)
               rects.append(Rect(x, y, winWidth, winHeight));
        return rects;
    }

    QList<Mat> getRandomNegs(const QList<Mat> &images) const
    {
        QList<Mat> negSamples;
        foreach (const Mat &image, images) {
            QList<int> xs = Common::RandSample(negsPerImage, image.cols - winWidth - 1, 0, true);
            QList<int> ys = Common::RandSample(negsPerImage, image.rows - winHeight - 1, 0, true);
            for (int i = 0; i < negsPerImage; i++)
                negSamples.append(image(Rect(xs[i], ys[i], winWidth, winHeight)));
        }
        return negSamples;
    }
};

BR_REGISTER(Transform, SlidingWindowTransform)

class ScaleImageTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(int winWidth READ get_winWidth WRITE set_winWidth RESET reset_winWidth STORED false)
    Q_PROPERTY(int winHeight READ get_winHeight WRITE set_winHeight RESET reset_winHeight STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(int maxSize READ get_maxSize WRITE set_maxSize RESET reset_maxSize STORED false)
    BR_PROPERTY(float, scaleFactor, 1.2)
    BR_PROPERTY(int, winWidth, 36)
    BR_PROPERTY(int, winHeight, 36)
    BR_PROPERTY(int, minSize, 12)
    BR_PROPERTY(int, maxSize, -1)

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        QList<float> scales;
        foreach (const Mat &m, src) {
            int imgWidth = m.cols, imgHeight = m.rows;

            int effectiveMaxSize = maxSize;
            if (effectiveMaxSize < 0) effectiveMaxSize = qMax(imgWidth, imgHeight);

            for (float factor = 1; ; factor *= scaleFactor) {
                int effectiveWinWidth = winWidth * factor, effectiveWinHeight = winHeight * factor;
                int scaledImgWidth = imgWidth / factor, scaledImgHeight = imgHeight / factor;

                if (scaledImgWidth < winWidth || scaledImgHeight < winHeight)
                    break;
                if (qMax(effectiveWinWidth, effectiveWinHeight) > effectiveMaxSize)
                    break;
                if (qMax(effectiveWinWidth, effectiveWinHeight) < minSize)
                    break;

                Mat scaledImg;
                resize(m, scaledImg, Size(scaledImgWidth, scaledImgHeight), 0, 0, CV_INTER_LINEAR);

                scales.append(factor);
                dst.append(scaledImg);
            }
        }
        dst.file.setList<float>("Scales", scales);
    }
};

BR_REGISTER(Transform, ScaleImageTransform)

/*!
 * \ingroup transforms
 * \brief Detects objects with OpenCV's built-in HOG detection.
 * \author Austin Blanton \cite imaus10
 */
class HOGDetectTransform : public UntrainableTransform
{
    Q_OBJECT

    HOGDescriptor hog;

    void init()
    {
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        std::vector<Rect> objLocs;
        QList<Rect> rects;
        hog.detectMultiScale(src, objLocs);
        foreach (const Rect &obj, objLocs)
            rects.append(obj);
        dst.file.setRects(rects);
    }
};

BR_REGISTER(Transform, HOGDetectTransform)

/*!
 * \ingroup transforms
 * \brief Consolidate redundant/overlapping detections.
 * \author Brendan Klare \cite bklare
 */
class ConsolidateDetectionsTransform : public Transform
{
    Q_OBJECT

public:
    ConsolidateDetectionsTransform() : Transform(false, false) {}
private:

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        if (!dst.file.contains("Confidences"))
            return;

        //Compute overlap between rectangles and create discrete Laplacian matrix
        QList<Rect> rects = OpenCVUtils::toRects(src.file.rects());
        int n = rects.size();
        if (n == 0)
            return;
        MatrixXf laplace(n,n);
        for (int i = 0; i < n; i++) {
            laplace(i,i) = 0;
        }
        for (int i = 0; i < n; i++){
            for (int j = i + 1; j < n; j++) {
                float overlap = (float)((rects[i] & rects[j]).area()) / (float)max(rects[i].area(), rects[j].area());
                if (overlap > 0.5) {
                    laplace(i,j) = -1.0;
                    laplace(j,i) = -1.0;
                    laplace(i,i) = laplace(i,i) + 1.0;
                    laplace(j,j) = laplace(j,j) + 1.0;
                } else {
                    laplace(i,j) = 0;
                    laplace(j,i) = 0;
                }
            }
        }

        // Compute eigendecomposition
        SelfAdjointEigenSolver<Eigen::MatrixXf> eSolver(laplace);
        MatrixXf allEVals = eSolver.eigenvalues();
        MatrixXf allEVecs = eSolver.eigenvectors();

        //Keep eigenvectors with zero eigenvalues
        int nRegions = 0;
        for (int i = 0; i < n; i++) {
            if (fabs(allEVals(i)) < 1e-4) {
                nRegions++;
            }
        }
        MatrixXf regionVecs(n, nRegions);
        for (int i = 0, cnt = 0; i < n; i++) {
            if (fabs(allEVals(i)) < 1e-4)
                regionVecs.col(cnt++) = allEVecs.col(i);
        }

        //Determine membership for each consolidated location
        // and compute average of regions. This is determined by
        // finding which eigenvector has the highest magnitude for
        // each input dimension. Each input dimension corresponds to
        // one of the input rect region. Thus, each eigenvector represents
        // a set of overlaping regions.
        float *midX = new float[nRegions];
        float *midY = new float[nRegions];
        float *avgWidth = new float[nRegions];
        float *avgHeight = new float[nRegions];
        float *confs = new float[nRegions];
        int *cnts = new int[nRegions];
        int mx;
        int mxIdx;
        for (int i = 0 ; i < nRegions; i++) {
            midX[i] = 0;
            midY[i] = 0;
            avgWidth[i] = 0;
            avgHeight[i] = 0;
            confs[i] = 0;
            cnts[i] = 0;
        }

        QList<float> confidences = dst.file.getList<float>("Confidences");
        for (int i = 0; i < n; i++) {
            mx = 0.0;
            mxIdx = -1;

            for (int j = 0; j < nRegions; j++) {
                if (fabs(regionVecs(i,j)) > mx) {
                    mx = fabs(regionVecs(i,j));
                    mxIdx = j;
                }
            }

            Rect curRect = rects[i];
            midX[mxIdx] += ((float)curRect.x + (float)curRect.width  / 2.0);
            midY[mxIdx] += ((float)curRect.y + (float)curRect.height / 2.0);
            avgWidth[mxIdx]  += (float) curRect.width;
            avgHeight[mxIdx] += (float) curRect.height;
            confs[mxIdx] += confidences[i];
            cnts[mxIdx]++;
        }

        QList<Rect> consolidatedRects;
        QList<float> consolidatedConfidences;
        for (int i = 0; i < nRegions; i++) {
            float cntF = (float) cnts[i];
            if (cntF > 0) {
                int x = qRound((midX[i] / cntF) - (avgWidth[i] / cntF) / 2.0);
                int y = qRound((midY[i] / cntF) - (avgHeight[i] / cntF) / 2.0);
                int w = qRound(avgWidth[i] / cntF);
                int h = qRound(avgHeight[i] / cntF);
                consolidatedRects.append(Rect(x,y,w,h));
                consolidatedConfidences.append(confs[i] / cntF);
            }
        }

        delete [] midX;
        delete [] midY;
        delete [] avgWidth;
        delete [] avgHeight;
        delete [] confs;
        delete [] cnts;

        dst.file.setRects(consolidatedRects);
        dst.file.setList<float>("Confidences", consolidatedConfidences);
    }
};

BR_REGISTER(Transform, ConsolidateDetectionsTransform)

class ExpandDetectionsTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach (const Template &t, src) {
            if (!t.file.contains("Confidences"))
                qFatal("Your rects need confidences");

            QList<float> confidences = t.file.getList<float>("Confidences");
            QList<QRectF> rects = t.file.rects();

            if (confidences.size() != rects.size())
                qFatal("The number of confidence values differs from the number of rects. Uh oh!");

            for (int i = 0; i < rects.size(); i++) {
                Template u(t.file);
                u.file.remove("Confidences");
                u.file.set("Confidence", confidences[i]);
                u.file.clearRects();
                u.file.appendRect(rects[i]);
                u.file.set("Rect", rects[i]);
                dst.append(u);
            }
        }
    }
};

BR_REGISTER(Transform, ExpandDetectionsTransform)

/*!
 * \ingroup transforms
 * \brief For each rectangle bounding box in src, a new
 *      template is created.
 * \author Brendan Klare \cite bklare
 */
class RectsToTemplatesTransform : public UntrainableMetaTransform
{
    Q_OBJECT

private:
    void project(const Template &src, Template &dst) const
    {
        Template tOut(src.file);
        QList<float> confidences = src.file.getList<float>("Confidences");
        QList<QRectF> rects = src.file.rects();
        for (int i = 0; i < rects.size(); i++) {
            Mat m(src, OpenCVUtils::toRect(rects[i]));
            Template t(src.file, m);
            t.file.set("Confidence", confidences[i]);
            t.file.clearRects();
            tOut << t;
        }
        dst = tOut;
    }
};

BR_REGISTER(Transform, RectsToTemplatesTransform)


} // namespace br

#include "slidingwindow.moc"
