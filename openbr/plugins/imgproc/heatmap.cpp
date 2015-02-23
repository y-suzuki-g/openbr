#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

class HeatmapTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(float alpha READ get_alpha WRITE set_alpha RESET reset_alpha STORED false)
    BR_PROPERTY(float, alpha, .5)

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        const float beta = 1-alpha;

        Mat buffer;
        applyColorMap(src, buffer, COLORMAP_JET);

        Mat colored;
        cvtColor(src, colored, CV_GRAY2BGR);

        addWeighted(colored,alpha,buffer,beta,0.0,dst);
    }
};

BR_REGISTER(Transform, HeatmapTransform)

} // namespace br

#include "imgproc/heatmap.moc"
