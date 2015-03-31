#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

class RndTransformTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(QList<int> rotationRange READ get_rotationRange WRITE set_rotationRange RESET reset_rotationRange STORED false)
    Q_PROPERTY(int maxTranslation READ get_maxTranslation WRITE set_maxTranslation RESET reset_maxTranslation STORED false)
    Q_PROPERTY(float maxScaleFactor READ get_maxScaleFactor WRITE set_maxScaleFactor RESET reset_maxScaleFactor STORED false)
    Q_PROPERTY(float flipProb READ get_flipProb WRITE set_flipProb RESET reset_flipProb STORED false)
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    BR_PROPERTY(QList<int>, rotationRange, QList<int>() << -15 << 15)
    BR_PROPERTY(int, maxTranslation, 0)
    BR_PROPERTY(float, maxScaleFactor, 0.1)
    BR_PROPERTY(float, flipProb, 0.0)
    BR_PROPERTY(int, n, 1)

    void project(const Template &src, Template &dst) const
    {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach (const Template &t, src) {
            foreach (const QRectF &rect, t.file.rects()) {
                for (int i = 0; i < n; i++) {
                    QRectF scaledRect = scaleRect(rect);
                    QRectF translatedRect = translateRect(scaledRect);
                    Mat rotatedImg = rotateImage(t, translatedRect.center());

                    Template u(t.file, rotatedImg(OpenCVUtils::toRect(translatedRect)));
                    u.file.setRects(QList<QRectF>() << translatedRect);
                    dst.append(u);
                }
            }
        }
    }

    QRectF scaleRect(const QRectF &rect) const
    {
        int percent = (int)maxScaleFactor * 100;
        float factor = (rand() % percent) / RAND_MAX;
        int coeff = (rand() % 2) - 1;
        float deltaW = coeff * rect.width() * factor;
        float deltaH = coeff * rect.height() * factor;
        return QRectF(rect.x() + deltaW, rect.y() + deltaH, rect.width() + 2*deltaW, rect.height() + 2*deltaH);
    }

    QRectF translateRect(const QRectF &rect) const
    {
        int dx = (rand() % (2 * maxTranslation)) - maxTranslation;
        int dy = (rand() % (2 * maxTranslation)) - maxTranslation;
        QRectF translatedRect = rect;
        translatedRect.translate(dx, dy);
        return translatedRect;
    }

    Mat rotateImage(const Mat &m, const QPointF &center) const
    {
        int span = rotationRange.first() - rotationRange.last();
        int angle = (rand() % span) + rotationRange.first();
        Mat rotMatrix = getRotationMatrix2D(OpenCVUtils::toPoint(center),angle,1.0);

        Mat out;
        warpAffine(m,out,rotMatrix,Size(m.cols,m.rows));
        return out;
    }
};

BR_REGISTER(Transform, RndTransformTransform)

} // namespace br

#include "imgproc/rndtransform.moc"
