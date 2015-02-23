#include <opencv2/objdetect/objdetect.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace std;
using namespace cv;

namespace br
{

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

} // namespace br

#include "metadata/nonmaxsuppression.moc"
