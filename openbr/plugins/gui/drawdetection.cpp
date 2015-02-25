#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

class DrawDetectionTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int lineThickness READ get_lineThickness WRITE set_lineThickness RESET reset_lineThickness STORED false)
    BR_PROPERTY(int, lineThickness, 1)

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        const QList<Rect> rects = OpenCVUtils::toRects(src.file.rects());
        const QList<float> confidences = src.file.getList<float>("Confidences");

        if (rects.size() != confidences.size())
            qFatal("Each detection needs a confidence score!");

        for (int i = 0; i < rects.size(); i++) {
            rectangle(dst, rects[i], Scalar(0, 255, 0), lineThickness);
            Point textPoint = rects[i].tl(); textPoint.x += 2; textPoint.y += 10;
            putText(dst, QString::number(confidences[i], 'g', 3).toStdString(), textPoint, FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 0, 255), 1);
        }
    }
};

BR_REGISTER(Transform, DrawDetectionTransform)

} // namespace br

#include "gui/drawdetection.moc"
