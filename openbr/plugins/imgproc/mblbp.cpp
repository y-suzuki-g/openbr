#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Shengcai Liao, Xiangxin Zhu, Zhen Lei, Lun Zhang, and Stan Z. Li;
 * "Learning Multi-scale Block Local Binary Patterns for Face Recognition"
 * \author Jordan Cheney \cite JordanCheney
 */
class MB_LBPRepresentation : public Representation
{
    Q_OBJECT

    Q_PROPERTY(int winWidth READ get_winWidth WRITE set_winWidth RESET reset_winWidth STORED false)
    Q_PROPERTY(int winHeight READ get_winHeight WRITE set_winHeight RESET reset_winHeight STORED false)
    BR_PROPERTY(int, winWidth, 24)
    BR_PROPERTY(int, winHeight, 24)

    void init()
    {
        for (int x = 0; x < winWidth; x++)
            for (int y = 0; y < winHeight; y++)
                for (int w = 1; w <= winWidth / 3; w++)
                    for (int h = 1; h <= winHeight / 3; h++)
                        if ((x+3*w <= winWidth) && (y+3*h <= winHeight))
                            features.append(Feature(x, y, w, h ));
    }

    Mat preprocess(const Mat &image) const
    {
        Mat in;
        integral(image, in);
        return in;
    }

    Mat evaluate(const Mat &image, const QList<int> &indices) const
    {
        if (image.rows != winHeight || image.cols != winWidth)
            qFatal("the image does not match the representation size");

        Mat results(1, indices.empty() ? features.size() : indices.size(), CV_32FC1);
        for (int idx = 0; idx < (indices.empty() ? features.size() : indices.size()); idx++)
            results.at<float>(idx) = indices.empty() ? features[idx].calc(image) : features[indices[idx]].calc(image);

        return results;
    }

    int numFeatures() const
    {
        return features.size();
    }

    struct Feature
    {
        Feature(int x, int y, int w, int h)
        {
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    pts[4*r + c] = Point(x + c*w, y + r*h);
        }

        float help(const Mat &img, int idx) const
        {
            return img.at<float>(pts[idx].y, pts[idx].x);
        }

        uchar calc(const Mat &img) const
        {
            float cval = help(img, 10) - help(img, 6) - help(img, 9) + help(img, 5);

            return (uchar)
                ((help(img, 5) - help(img, 4) - help(img, 1) + help(img, 0) >= cval ? 128 : 0) |   // 0
                 (help(img, 6) - help(img, 5) - help(img, 2) + help(img, 1) >= cval ? 64 : 0) |    // 1
                 (help(img, 7) - help(img, 6) - help(img, 3) + help(img, 2) >= cval ? 32 : 0) |    // 2
                 (help(img, 11) - help(img, 10) - help(img, 7) + help(img, 6) >= cval ? 16 : 0) |  // 5
                 (help(img, 15) - help(img, 14) - help(img, 11) + help(img, 10) >= cval ? 8 : 0) | // 8
                 (help(img, 14) - help(img, 13) - help(img, 10) + help(img, 9) >= cval ? 4 : 0) |  // 7
                 (help(img, 13) - help(img, 12) - help(img, 9) + help(img, 8) >= cval ? 2 : 0) |   // 6
                 (help(img, 9) - help(img, 8) - help(img, 5) + help(img, 4) >= cval ? 1 : 0));     // 3
        }
        Point pts[16];
    };
    QList<Feature> features;
};

BR_REGISTER(Representation, MB_LBPRepresentation)

} // namespace br

#include "imgproc/mblbp.moc"
