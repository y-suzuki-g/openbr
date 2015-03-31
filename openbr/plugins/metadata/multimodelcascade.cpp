#include <QDirIterator>
#include <opencv2/objdetect/objdetect.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

class CascadeResourceMaker : public ResourceMaker<QList<CascadeClassifier*> >
{
    QStringList files;

public:
    CascadeResourceMaker(const QString &modelDir)
    {
        QDirIterator it(modelDir);
        while(it.hasNext())
            files.append(it.next());
    }

private:
    QList<CascadeClassifier*> *make() const
    {
        QList<CascadeClassifier*> *cascades = new QList<CascadeClassifier*>();
        foreach (const QString &file, files) {
            CascadeClassifier *cascade = new CascadeClassifier();
            if (!cascade->load(file.toStdString()))
                qFatal("Failed to load: %s", qPrintable(file));
            cascades->append(cascade);
        }
        return cascades;
    }
};

class MultiModelCascadeTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(QString modelDir READ get_modelDir WRITE set_modelDir RESET reset_modelDir STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(int minNeighbors READ get_minNeighbors WRITE set_minNeighbors RESET reset_minNeighbors STORED false)
    Q_PROPERTY(bool ROCMode READ get_ROCMode WRITE set_ROCMode RESET reset_ROCMode STORED false)
    BR_PROPERTY(QString, modelDir, Globals->sdkPath + "/share/openbr/models/openbrcascades/")
    BR_PROPERTY(int, minSize, 64)
    BR_PROPERTY(float, scaleFactor, 1.2)
    BR_PROPERTY(int, minNeighbors, 5)
    BR_PROPERTY(bool, ROCMode, false)

    Resource<QList<CascadeClassifier*> > cascadeResource;

    void init()
    {
        cascadeResource.setResourceMaker(new CascadeResourceMaker(modelDir));
    }

    void project(const Template &src, Template &dst) const
    {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        QList<CascadeClassifier*> *cascades = cascadeResource.acquire();

        foreach (const Template &t, src) {
            const bool enrollAll = t.file.getBool("enrollAll");

            // Mirror the behavior of ExpandTransform in the special case
            // of an empty template.
            if (t.empty() && !enrollAll) {
                dst.append(t);
                continue;
            }

            for (int i=0; i<t.size(); i++) {
                Mat m;
                OpenCVUtils::cvtUChar(t[i], m);

                std::vector<Rect> rects;
                std::vector<int> rejectLevels;
                std::vector<double> levelWeights;
                foreach (CascadeClassifier *cascade, *cascades) {
                    std::vector<Rect> cascadeRects;
                    std::vector<int> cascadeRejectLevels;
                    std::vector<double> cascadeLevelWeights;
                    if (ROCMode) cascade->detectMultiScale(m, cascadeRects, cascadeRejectLevels, cascadeLevelWeights, scaleFactor, minNeighbors, (enrollAll ? 0 : CASCADE_FIND_BIGGEST_OBJECT) | CASCADE_SCALE_IMAGE, Size(minSize, minSize), Size(), true);
                    else         cascade->detectMultiScale(m, cascadeRects, scaleFactor, minNeighbors, enrollAll ? 0 : CASCADE_FIND_BIGGEST_OBJECT, Size(minSize, minSize));

                    rects.insert(rects.end(), cascadeRects.begin(), cascadeRects.end());
                    rejectLevels.insert(rejectLevels.end(), cascadeRejectLevels.begin(), cascadeRejectLevels.end());
                    levelWeights.insert(levelWeights.end(), cascadeLevelWeights.begin(), cascadeLevelWeights.end());
                }

                groupRectangles(rects, rejectLevels, levelWeights, minNeighbors);

                if (!enrollAll && rects.empty())
                    rects.push_back(Rect(0, 0, m.cols, m.rows));

                for (size_t j = 0; j < rects.size(); j++) {
                    Template u(t.file, m);
                    if (rejectLevels.size() > j) u.file.set("Confidence", rejectLevels[j]*levelWeights[j]);
                    else                         u.file.set("Confidence", 1);

                    const QRectF rect = OpenCVUtils::fromRect(rects[j]);
                    u.file.appendRect(rect);
                    u.file.set("Face", rect);
                    dst.append(u);
                }
            }
        }

        cascadeResource.release(cascades);
    }
};

} // namespace br

#include "metadata/multimodelcascade.moc"
