#include <openbr/plugins/openbr_internal.h>

namespace br
{

class RegisterPointsAsAffine : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<int> pointsIdxs READ get_pointIdxs WRITE set_pointIdxs RESET reset_pointIdxs STORED false)
    BR_PROPERTY(QList<int>, pointIdxs, QList<int>())

    void projectMetadata(const File &src, File &dst) const
    {
        if (pointIdxs.size() != 2 && pointIdxs.size() != 3)
            qFatal("Need 2 or 3 points for affine transform");

        dst = src;

        QList<QPointF> points = src.points();

        dst.set("Affine_0", points[pointIdxs[0]]);
        dst.set("Affine_1", points[pointIdxs[1]]);
        if (pointIdxs.size() == 3)
            dst.set("Affine_2", points[pointIdxs[2]]);
    }
};

BR_REGISTER(Transform, RegisterPointsAsAffine)

} // namespace br

#include "metadata/registerpointsasaffine.moc"
