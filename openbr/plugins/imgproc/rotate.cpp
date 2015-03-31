#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

class RotateTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(int angle READ get_angle WRITE set_angle RESET reset_angle STORED false)
    BR_PROPERTY(int, angle, 0)

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.file.set("Angle", angle);

        Mat rotMatrix = getRotationMatrix2D(Point2f(src.m().rows/2,src.m().cols/2), angle, 1.0);

        Mat u;
        warpAffine(src, u, rotMatrix, Size(src.m().cols,src.m().rows), INTER_LINEAR, BORDER_REFLECT_101);
        dst.m() = u;
    }
};

BR_REGISTER(Transform, RotateTransform)

} // namespace br

#include "imgproc/rotate.moc"
