#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

class BootstrapTransform : public Transform
{
    Q_OBJECT

    Q_PROPERTY(br::Transform *sampler READ get_sampler WRITE set_sampler RESET reset_sampler STORED false)
    Q_PROPERTY(br::Transform *classifier READ get_classifier WRITE set_classifier RESET reset_classifier STORED false)
    Q_PROPERTY(int winWidth READ get_winWidth WRITE set_winWidth RESET reset_winWidth STORED false)
    Q_PROPERTY(int winHeight READ get_winHeight WRITE set_winHeight RESET reset_winHeight STORED false)
    Q_PROPERTY(int iterations READ get_iterations WRITE set_iterations RESET reset_iterations STORED false)
    BR_PROPERTY(br::Transform*, sampler, NULL)
    BR_PROPERTY(br::Transform*, classifier, NULL)
    BR_PROPERTY(int, winWidth, 24)
    BR_PROPERTY(int, winHeight, 24)
    BR_PROPERTY(int, iterations, 5)

    void train(const TemplateList &data)
    {
        TemplateList mutData = data, bsData;
        foreach (const Template &src, data) {
            Template dst;
            sampler->project(src, dst);
            bsData.append(dst);
        }

        classifier->train(bsData);
        bsData.clear();

        for (int it = 0; it < iterations; it++) {
            for (int i = 0; i < mutData.size(); i++) {
                Template t = mutData[i];
                QList<Rect> gtRects = OpenCVUtils::toRects(t.file.rects());
                t.file.clearRects();

                Template dst;
                classifier->project(t, dst);

                hardSamples(dst, bsData, gtRects);
            }

            classifier->train(bsData);
            bsData.clear();
        }
    }

    void project(const Template &src, Template &dst) const
    {
        classifier->project(src, dst);
    }

    void hardSamples(const Template &t, TemplateList &data, QList<Rect> &gtRects)
    {
        QList<Rect> predRects = OpenCVUtils::toRects(t.file.rects());

        // first get the positive samples
        foreach (const Rect &r, gtRects) {
            Rect safe_r(qMax(r.x, 0), qMax(r.y, 0), qMin(t.m().cols - qMax(r.x, 0), r.width), qMin(t.m().rows - qMax(r.y, 0), r.height));
            Mat pos;
            resize(t.m()(safe_r), pos, Size(winWidth, winHeight));
            Template u(t.file, pos);
            u.file.set("Label", QVariant::fromValue(1.));
            data.append(u);
        }

        // now the hard negatives
        foreach (const Rect &pr, predRects) {
            if (OpenCVUtils::overlaps(gtRects, pr, 0.5))
                continue;

            Mat neg;
            resize(t.m()(pr), neg, Size(winWidth, winHeight));
            Template u(t.file, neg);
            u.file.set("Label", QVariant::fromValue(0.));
            data.append(u);
        }
    }
};

} // namespace br

#include "metadata/bootstrap.moc"
