#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

class RotateCopyTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<int> angles READ get_angles WRITE set_angles RESET reset_angles STORED false)
    BR_PROPERTY(QList<int>, angles, QList<int>())

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        foreach (int angle, angles) {
            Mat rotMatrix = getRotationMatrix2D(Point2f(src.m().rows/2,src.m().cols/2), angle, 1.0);

            Mat u;
            warpAffine(src, u, rotMatrix, Size(src.m().cols,src.m().rows), INTER_LINEAR, BORDER_REFLECT_101);
            dst.append(u);
        }
        dst.file.setList<int>("Angles", QList<int>() << 0 << angles);
        qDebug("dst: %d", dst.size());
    }
};

BR_REGISTER(Transform, RotateCopyTransform)

} // namespace br

#include "imgproc/rotatecopy.moc"
