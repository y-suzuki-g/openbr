#include <opencv2/ml/ml.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup representations
 * \brief Wraps OpenCV's multi-layer perceptron framework
 * \author Scott Klum \cite sklum
 * \author Jordan Cheney \cite JordanCheney
 * \brief http://docs.opencv.org/modules/ml/doc/neural_networks.html
 */
class MLPRepresentation : public Representation
{
    Q_OBJECT

    Q_ENUMS(Kernel)

    Q_PROPERTY(Kernel kernel READ get_kernel WRITE set_kernel RESET reset_kernel STORED false)
    Q_PROPERTY(float alpha READ get_alpha WRITE set_alpha RESET reset_alpha STORED false)
    Q_PROPERTY(float beta READ get_beta WRITE set_beta RESET reset_beta STORED false)
    Q_PROPERTY(QList<int> neuronsPerLayer READ get_neuronsPerLayer WRITE set_neuronsPerLayer RESET reset_neuronsPerLayer STORED false)

public:

    enum Kernel { Identity = CvANN_MLP::IDENTITY,
                  Sigmoid = CvANN_MLP::SIGMOID_SYM,
                  Gaussian = CvANN_MLP::GAUSSIAN};

private:
    BR_PROPERTY(Kernel, kernel, Sigmoid)
    BR_PROPERTY(float, alpha, 1)
    BR_PROPERTY(float, beta, 1)
    BR_PROPERTY(QList<int>, neuronsPerLayer, QList<int>())

    CvANN_MLP mlp;

    void init()
    {
        if (kernel == Gaussian)
            qWarning("The OpenCV documentation warns that the Gaussian kernel, \"is not completely supported at the moment\"");

        if (neuronsPerLayer.size() < 2)
            qFatal("Net needs at least 2 layers!");

        Mat layers = OpenCVUtils::toMat(neuronsPerLayer);
        mlp.create(layers, kernel, alpha, beta);
    }

    void train(const QList<Mat> &_images, const QList<float> &_labels)
    {
        foreach(const Mat &image, _images)
            if ((int)image.total() != neuronsPerLayer.first())
                qFatal("The given image has more pixels then the first layer of the net!");

        if (_labels.size() != (_images.size() * neuronsPerLayer.last()))
            qFatal("The number of given labels should be equal to num_images * last_layer_neurons");

        Mat data = OpenCVUtils::toMat(_images);
        Mat labels = OpenCVUtils::toMat(_labels, _images.size());

        mlp.train(data,labels,Mat());
    }

    Mat evaluate(const Mat &image, const QList<int> &indices) const
    {
        if (!indices.empty())
            qFatal("MLP does not have feature indices!");

        Mat response(neuronsPerLayer.size(), 1, CV_32FC1);
        mlp.predict(image.reshape(1, 1), response);
        return response;
    }

    Size windowSize() const
    {
        // right now lets assume square input
        return Size((int)sqrt(neuronsPerLayer.first()), (int)sqrt(neuronsPerLayer.first()));
    }

    int numFeatures() const
    {
        return neuronsPerLayer.last();
    }

    void load(QDataStream &stream)
    {
        OpenCVUtils::loadModel(mlp, stream);
    }

    void store(QDataStream &stream) const
    {
        OpenCVUtils::storeModel(mlp, stream);
    }
};

BR_REGISTER(Representation, MLPRepresentation)

} // namespace br

#include "representation/mlp.moc"
