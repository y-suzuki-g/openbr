#include <numeric>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>
#include <openbr/core/opencvutils.h>
#include <openbr/core/qtutils.h>

using namespace cv;

namespace br
{

class RndRectTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(int padding READ get_padding WRITE set_padding RESET reset_padding STORED false)
    Q_PROPERTY(int sampleFactor READ get_sampleFactor WRITE set_sampleFactor RESET reset_sampleFactor)
    Q_PROPERTY(bool duplicatePositiveSamples READ get_duplicatePositiveSamples WRITE set_duplicatePositiveSamples RESET reset_duplicatePositiveSamples STORED true)
    Q_PROPERTY(QList<float> sampleOverlapBands READ get_sampleOverlapBands WRITE set_sampleOverlapBands RESET reset_sampleOverlapBands STORED true)
    Q_PROPERTY(int samplesPerOverlapBand READ get_samplesPerOverlapBand WRITE set_samplesPerOverlapBand RESET reset_samplesPerOverlapBand STORED true)
    Q_PROPERTY(bool classification READ get_classification WRITE set_classification RESET reset_classification STORED true)
    Q_PROPERTY(float overlapPower READ get_overlapPower WRITE set_overlapPower RESET reset_overlapPower STORED true)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED true)
    BR_PROPERTY(int, padding, 0)
    BR_PROPERTY(int, sampleFactor, 4)
    BR_PROPERTY(bool, duplicatePositiveSamples, true)
    BR_PROPERTY(QList<float>, sampleOverlapBands, QList<float>() << .1 << .5 << .7 << .9 << 1.0)
    BR_PROPERTY(int, samplesPerOverlapBand, 2)
    BR_PROPERTY(bool, classification, true)
    BR_PROPERTY(float, overlapPower, 1)
    BR_PROPERTY(QString, inputVariable, "Label")

    void project(const Template &src, Template &dst) const {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach(const Template &t, src) {
            QList<QRectF> rects = t.file.rects();
            foreach (const QRectF rect, rects) {
                QRectF region(rect.x() - padding, rect.y() - padding, rect.width() + 2*padding, rect.height() + 2*padding);

                if (region.x() < 0 ||
                    region.y() < 0 ||
                    region.x() + region.width() >= t.m().cols ||
                    region.y() + region.height() >= t.m().rows)
                    continue;

                int positiveSamples = duplicatePositiveSamples ? (sampleOverlapBands.size()-1)*samplesPerOverlapBand : 1;
                for (int k=0; k<positiveSamples; k++) {
                    dst.append(Template(t.file, Mat(t.m(), OpenCVUtils::toRect(region))));
                    dst.last().file.set(inputVariable, 1);
                }

                QList<int> labelCount;
                for (int k=0; k<sampleOverlapBands.size()-1; k++)
                    labelCount << 0;

                while (std::accumulate(labelCount.begin(),labelCount.end(),0.0) < (sampleOverlapBands.size()-1)*samplesPerOverlapBand) {
                    int x = Common::RandSample(1, region.x() + sampleFactor*region.width(), region.x() - sampleFactor/(2*region.width()))[0];
                    int y = Common::RandSample(1, region.y() + sampleFactor*region.height(), region.y() - sampleFactor/(2*region.height()))[0];
                    int w = Common::RandSample(1, sampleFactor*region.width(), region.width())[0];
                    int h = Common::RandSample(1, sampleFactor*region.height(), region.height())[0];

                    if (x < 0 || y < 0 || x + w > t.m().cols || y + h > t.m().rows)
                        continue;

                    QRectF negativeLocation = QRectF(x, y, w, h);

                    float overlap = pow(QtUtils::overlap(region, negativeLocation),overlapPower);

                    for (int k = 0; k<sampleOverlapBands.size()-1; k++) {
                        if (overlap >= sampleOverlapBands.at(k) && overlap < sampleOverlapBands.at(k+1) && labelCount[k] < samplesPerOverlapBand) {
                            Mat m(t.m(),OpenCVUtils::toRect(negativeLocation));
                            dst.append(Template(t.file,  m));
                            float label = classification ? 0 : overlap;
                            dst.last().file.set(inputVariable, label);
                            labelCount[k]++;
                        }
                    }
                }
            }
        }
    }
};

BR_REGISTER(Transform, RndRectTransform)

} // namespace br

#include "imgproc/rndrect.moc"
