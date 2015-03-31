#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

class RotateCopyTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<int> angles READ get_angles WRITE set_angles RESET reset_angles STORED false)
    BR_PROPERTY(QList<int>, angles, QList<int>())

    void project(const Template &src, Template &dst) const
    {
        TemplateList temp;
        project(TemplateList() << src, temp);
        dst = temp.first();
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        if (src.empty())
            return;

        const Template t = src.first();

        dst.append(t);

        foreach (int angle, angles) {
            Mat rotMatrix = getRotationMatrix2D(Point2f(t.m().rows/2,t.m().cols/2), angle, 1.0);

            Template u(t.file);
            u.file.set("Angle", angle);

            warpAffine(src.first(), u, rotMatrix, Size(t.m().cols,t.m().rows), INTER_LINEAR, BORDER_REFLECT_101);
            dst.append(u);
        }
    }
};

BR_REGISTER(Transform, RotateCopyTransform)

} // namespace br

#include "imgproc/rotatecopy.moc"
