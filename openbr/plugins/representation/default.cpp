#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

class DefaultRepresentation : public Representation
{
    Q_OBJECT
    Q_PROPERTY(int winWidth READ get_winWidth WRITE set_winWidth RESET reset_winWidth STORED false)
    Q_PROPERTY(int winHeight READ get_winHeight WRITE set_winHeight RESET reset_winHeight STORED false)
    BR_PROPERTY(int, winWidth, 24)
    BR_PROPERTY(int, winHeight, 24)

    Mat evaluate(const Mat &image, const QList<int> &indices) const
    {
        (void) indices;
        return image;
    }

    Size windowSize() const
    {
        return Size(winWidth, winHeight);
    }

    int numFeatures() const
    {
        return winWidth * winHeight;
    }
};

BR_REGISTER(Representation, DefaultRepresentation)

} // namespace br

#include "representation/default.moc"
