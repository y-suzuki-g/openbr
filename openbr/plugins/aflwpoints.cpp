#include <openbr/plugins/openbr_internal.h>

namespace br
{

class AFLWPointsTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        const int lBrow = 2;
        const int rBrow = 3;
        const int chin = 20;
        const int LELC = 6;
        const int RERC = 11;

        QList<QPointF> points = src.file.points();

        QPointF eyeCenter = (points[lBrow] + points[rBrow]) / 2;

        int dx = eyeCenter.x() - points[chin].x(), dy = eyeCenter.y() - points[chin].y();
        QPointF forehead(safe(eyeCenter.x() + 0.4*dx, 1, src.m().cols),
                         safe(eyeCenter.y() + 0.4*dy, 1, src.m().rows));

        QPointF left(points[LELC]);
        QPointF right(points[RERC]);
        int distance = qMax(eyeCenter.x() - left.x(), right.x() - eyeCenter.x());
        distance += 0.2*distance;
        left.setX(safe(eyeCenter.x() - distance, 1, src.m().cols));
        right.setX(safe(eyeCenter.x() + distance, 1, src.m().cols));

        dst.file.setPoints(QList<QPointF>() << eyeCenter << forehead << points[chin] << left << right);
    }

    static inline float safe(float val, float min, float max)
    {
        return qMin(max, qMax(min, val));
    }
};

BR_REGISTER(Transform, AFLWPointsTransform)

} // namespace br

#include "metadata/aflwpoints.moc"
