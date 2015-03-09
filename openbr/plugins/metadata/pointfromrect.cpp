#include <openbr/plugins/openbr_internal.h>

namespace br
{

class PointFromRectTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;

        QList<QRectF> rects = src.rects();
        foreach (QRectF rect, rects)
            dst.appendPoint(rect.center());
    }
};

BR_REGISTER(Transform, PointFromRectTransform)

} // namespace br

#include "metadata/pointfromrect.moc"
