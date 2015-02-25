/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

class SlidingWindowTransform : public Transform
{
    Q_OBJECT

    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform STORED false)
    Q_PROPERTY(int winWidth READ get_winWidth WRITE set_winWidth RESET reset_winWidth STORED false)
    Q_PROPERTY(int winHeight READ get_winHeight WRITE set_winHeight RESET reset_winHeight STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(int maxSize READ get_maxSize WRITE set_maxSize RESET reset_maxSize STORED false)
    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(float threshold READ get_threshold WRITE set_threshold RESET reset_threshold STORED false)

    Q_PROPERTY(bool bootstrap READ get_bootstrap WRITE set_bootstrap RESET reset_bootstrap STORED false)
    Q_PROPERTY(int numNegs READ get_numNegs WRITE set_numNegs RESET reset_numNegs STORED false)
    Q_PROPERTY(int iterations READ get_iterations WRITE set_iterations RESET reset_iterations STORED false)

    BR_PROPERTY(br::Transform*, transform, NULL)
    BR_PROPERTY(int, winWidth, 24)
    BR_PROPERTY(int, winHeight, 24)
    BR_PROPERTY(int, minSize, 24)
    BR_PROPERTY(int, maxSize, -1)
    BR_PROPERTY(float, scaleFactor, 1.2)
    BR_PROPERTY(float, threshold, 0)

    BR_PROPERTY(bool, bootstrap, false)
    BR_PROPERTY(int, numNegs, 5)
    BR_PROPERTY(int, iterations, 5)

    void train(const TemplateList &data)
    {
        if (!bootstrap) {
            transform->train(data);
            return;
        }

        TemplateList mutData = data, bsData;
        foreach (const Template &t, data)
            randomSamples(t, bsData);

        transform->train(bsData);
        bsData.clear();

        for (int it = 0; it < iterations; it++) {
            for (int i = 0; i < mutData.size(); i++) {
                Template t = mutData[i];
                QList<Rect> gtRects = OpenCVUtils::toRects(t.file.rects());
                t.file.clearRects();

                Template dst;
                project(t, dst);

                hardSamples(dst, bsData, gtRects);
            }

            transform->train(bsData);
            bsData.clear();
        }
    }

    void project(const Template &src, Template &dst) const
    {
        if (src.size() != 1)
            qFatal("Sliding Window only supports templates with 1 mat");

        dst = src;

        const Mat m = src.first();

        int effectiveMaxSize = maxSize;
        if (maxSize < 0)
            effectiveMaxSize = qMax(m.rows, m.cols);

        const float scaleFrom = qMin(winWidth/(float)effectiveMaxSize, winHeight/(float)effectiveMaxSize);
        const float scaleTo = qMax(winWidth/(float)minSize, winHeight/(float)minSize);

        Mat scaledImage; QList<float> confidences;
        for (float scale = scaleFrom; scale < scaleTo + 0.001; scale *= scaleFactor) {
            resize(m, scaledImage, Size(), scale, scale);

            const int step = scale < 1. ? 2 : 4;
            for (int y = 0; y < (scaledImage.rows - winHeight); y += step) {
                for (int x = 0; x < (scaledImage.cols - winWidth); x += step) {
                    Template u(src.file, scaledImage(Rect(x, y, winWidth, winHeight))), t;
                    transform->project(u, t);

                    float confidence = t.m().at<float>(0,0);
                    if (confidence > threshold) {
                        dst.file.appendRect(Rect(qRound(x/scale), qRound(y/scale), qRound(winWidth/scale), qRound(winHeight/scale)));
                        confidences.append(confidence);
                    }
                }
            }
            dst.file.setList<float>("Confidences", confidences);
        }
    }

    void randomSamples(const Template &t, TemplateList &data)
    {
        QList<Rect> gtRects = OpenCVUtils::toRects(t.file.rects());

        // first get the positive samples
        foreach (const Rect &r, gtRects) {
            Rect safe_r(qMax(r.x, 0), qMax(r.y, 0), qMin(t.m().cols - qMax(r.x, 0), r.width), qMin(t.m().rows - qMax(r.y, 0), r.height));
            Mat pos;
            resize(t.m()(safe_r), pos, Size(winWidth, winHeight));
            Template u(t.file, pos);
            u.file.set("Label", QVariant::fromValue(1.));
            data.append(u);
        }

        Common::seedRNG();

        // now the random negatives
        int negCount = 0;
        while (negCount < numNegs) {
            int x = Common::RandSample(1, t.m().cols - winWidth, 0).first();
            int y = Common::RandSample(1, t.m().rows - winHeight, 0).first();
            int w = Common::RandSample(1, t.m().cols - x, 5).first();
            int h = Common::RandSample(1, t.m().rows - y, 5).first();

            Rect nr(x, y, w, h);

            if (OpenCVUtils::overlaps(gtRects, nr, 0.5))
                continue;

            Mat neg;
            resize(t.m()(nr), neg, Size(winWidth, winHeight));
            Template u(t.file, neg);
            u.file.set("Label", QVariant::fromValue(0.));
            data.append(u);
            negCount++;
        }
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

BR_REGISTER(Transform, SlidingWindowTransform)

} // namespace br

#include "imgproc/slidingwindow.moc"
