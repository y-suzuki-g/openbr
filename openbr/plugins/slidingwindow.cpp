#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/common.h"
#include "openbr/core/qtutils.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace br
{

class SlidingWindowTransform : public Transform
{
    Q_OBJECT

    Q_PROPERTY(br::Classifier *classifier READ get_classifier WRITE set_classifier RESET reset_classifier STORED false)
    Q_PROPERTY(int winWidth READ get_winWidth WRITE set_winWidth RESET reset_winWidth STORED false)
    Q_PROPERTY(int winHeight READ get_winHeight WRITE set_winHeight RESET reset_winHeight STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(int maxSize READ get_maxSize WRITE set_maxSize RESET reset_maxSize STORED false)
    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(int negsPerImage READ get_negsPerImage WRITE set_negsPerImage RESET reset_negsPerImage STORED false)
    BR_PROPERTY(br::Classifier*, classifier, NULL)
    BR_PROPERTY(int, winWidth, 24)
    BR_PROPERTY(int, winHeight, 24)
    BR_PROPERTY(int, minSize, 24)
    BR_PROPERTY(int, maxSize, -1)
    BR_PROPERTY(float, scaleFactor, 1.2)
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

        // check if negImages are of the correct size or need to be randomly cropped
        if (negImages[0].rows != winHeight || negImages[0].cols != winWidth)
            getRandomCrops(negImages);

        // rough heuristic (numNegImages <= 3*numPosImages) to keep amount of negatives reasonable
        negImages = negImages.mid(0, qMin(3*posImages.size(), negImages.size()));

        QList<float> posLabels = QList<float>::fromVector(QVector<float>(posImages.size(), 1.0f));
        QList<float> negLabels = QList<float>::fromVector(QVector<float>(negImages.size(), 0.0f));

        QList<Mat> images = posImages; images.append(negImages);
        QList<float> labels = posLabels; labels.append(negLabels);
        classifier->train(images, labels);
    }

    void project(const Template &src, Template &dst) const
    {
        if (src.size() != 1)
            qFatal("Sliding Window only supports templates with 1 mat");

        dst = src;

        const Mat m = src.first();

        int effectiveMaxSize = maxSize;
        if (maxSize < 0)
            effectiveMaxSize = qMax(m.rows, m.cols);

        // preallocate buffer for scaled image
        Mat imageBuffer(m.rows, m.cols, CV_8U);

        QList<Rect> detections;
        QList<float> confidences;
        for (double factor = 1; ; factor *= scaleFactor) {
            Size scaledWindowSize(cvRound(winWidth*factor), cvRound(winHeight*factor));
            Size scaledImageSize(cvRound(m.cols/factor), cvRound(m.rows/factor));

            if (scaledImageSize.width <= winWidth || scaledImageSize.height <= winHeight)
                break;
            if (qMax(scaledWindowSize.width, scaledWindowSize.height) > effectiveMaxSize)
                break;
            if (qMin(scaledWindowSize.width, scaledWindowSize.height) < minSize)
                continue;

            Mat scaledImage(scaledImageSize, CV_8U, imageBuffer.data);
            resize(m, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR);

            const int step = factor > 2. ? 1 : 2;
            for (int y = 0; y < (scaledImage.rows - winHeight); y += step) {
                for (int x = 0; x < (scaledImage.cols - winWidth); x += step) {
                    float confidence = classifier->classify(scaledImage(Rect(x, y, winWidth, winHeight)));
                    if (confidence > 0) {
                        detections.append(Rect(cvRound(x*factor), cvRound(y*factor), cvRound(winWidth*factor), cvRound(winHeight*factor)));
                        confidences.append(confidence);
                    }
                }
            }
        }

        dst.file.appendRects(detections);
        dst.file.setList<float>("Confidences", confidences);
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
    void getRandomCrops(QList<Mat> &images) const
    {
        Common::seedRNG();
        QList<Mat> negSamples;
        foreach (const Mat &image, images) {
            QList<int> xs = Common::RandSample(negsPerImage, image.cols - winWidth - 1, 0, true);
            QList<int> ys = Common::RandSample(negsPerImage, image.rows - winHeight - 1, 0, true);
            for (int i = 0; i < negsPerImage; i++)
                negSamples.append(image(Rect(xs[i], ys[i], winWidth, winHeight)));
        }
        images.clear();
        images = negSamples;
    }
};

BR_REGISTER(Transform, SlidingWindowTransform)

class NonMaxSuppressionTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(float eps READ get_eps WRITE set_eps RESET reset_eps STORED false)
    Q_PROPERTY(int minNeighbors READ get_minNeighbors WRITE set_minNeighbors RESET reset_minNeighbors STORED false)
    BR_PROPERTY(float, eps, 0.2)
    BR_PROPERTY(int, minNeighbors, 5)

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        vector<Rect> detections = OpenCVUtils::toRects(src.file.rects()).toVector().toStdVector();
        vector<float> confidences = src.file.getList<float>("Confidences", QList<float>::fromVector(QVector<float>(detections.size(), 1.))).toVector().toStdVector();

        vector<int> labels;
        int nclasses = cv::partition(detections, labels, SimilarRects(eps));

        vector<Rect> rDetections(nclasses);
        vector<double> rConfidences(nclasses, DBL_MIN);
        vector<int> neighbors(nclasses, 0);

        // average class rectangles and take the best confidence
        for (int i = 0; i < (int)labels.size(); i++) {
            int cls = labels[i];
            rDetections[cls].x += detections[i].x;
            rDetections[cls].y += detections[i].y;
            rDetections[cls].width += detections[i].width;
            rDetections[cls].height += detections[i].height;
            if (rConfidences[cls] < confidences[i])
                rConfidences[cls] = confidences[i];
            neighbors[cls]++;
        }

        for (int i = 0; i < nclasses; i++) {
            Rect r = rDetections[i];
            float s = 1.f/neighbors[i];
            rDetections[i] = Rect(saturate_cast<int>(r.x*s), saturate_cast<int>(r.y*s),
                               saturate_cast<int>(r.width*s), saturate_cast<int>(r.height*s));
        }

        detections.clear();
        confidences.clear();

        for (int i = 0; i < nclasses; i++) {
            Rect r1 = rDetections[i];
            int n1 = neighbors[i];

            if (n1 <= minNeighbors)
                continue;

            // filter out small face rectangles inside large rectangles
            int j;
            for (j = 0; j < nclasses; j++) {
                int n2 = neighbors[j];

                if (j == i || n2 <= minNeighbors)
                    continue;
                Rect r2 = rDetections[j];

                int dx = saturate_cast<int>( r2.width * eps );
                int dy = saturate_cast<int>( r2.height * eps );

                if( i != j && r1.x >= r2.x - dx && r1.x + r1.width <= r2.x + r2.width + dx &&
                              r1.y >= r2.y - dy && r1.y + r1.height <= r2.y + r2.height + dy)
                    break;
            }

            if (j == nclasses) {
                detections.push_back(r1);
                confidences.push_back(rConfidences[i]);
            }
        }

        dst.file.setRects(QList<Rect>::fromVector(QVector<Rect>::fromStdVector(detections)));
        dst.file.setList<float>("Confidences", QList<float>::fromVector(QVector<float>::fromStdVector(confidences)));
    }
};

BR_REGISTER(Transform, NonMaxSuppressionTransform)

class ExpandDetectionsTransform : public UntrainableMetaTransform
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
            const QList<QRectF> rects = t.file.rects();
            const QList<float> confidences = t.file.getList<float>("Confidences");
            for (int i = 0; i < rects.size(); i++) {
                Template u(t.file, t.m());
                u.file.setRects(QList<QRectF>() << rects[i]);
                u.file.set("Rect", rects[i]);
                u.file.set("Confidence", confidences[i]);
                u.file.remove("Confidences");
                dst.append(u);
            }
        }
    }
};

BR_REGISTER(Transform, ExpandDetectionsTransform)

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
