#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Expand a template with multiple detections to multiple templates each with a single detection.
 * A detection in this case is defined as a rectangle with an associated confidence. The confidence
 * should be stored in the metadata variable "Confidences"
 * \author Jordan Cheney \cite JordanCheney
 */
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

} // namespace br

#include "core/expanddetections.moc"
