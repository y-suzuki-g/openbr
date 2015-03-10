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

class SlidingWindowTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform STORED false)
    Q_PROPERTY(int winWidth READ get_winWidth WRITE set_winWidth RESET reset_winWidth STORED false)
    Q_PROPERTY(int winHeight READ get_winHeight WRITE set_winHeight RESET reset_winHeight STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(int maxSize READ get_maxSize WRITE set_maxSize RESET reset_maxSize STORED false)
    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(float threshold READ get_threshold WRITE set_threshold RESET reset_threshold STORED false)
    BR_PROPERTY(br::Transform*, transform, NULL)
    BR_PROPERTY(int, winWidth, 24)
    BR_PROPERTY(int, winHeight, 24)
    BR_PROPERTY(int, minSize, 24)
    BR_PROPERTY(int, maxSize, -1)
    BR_PROPERTY(float, scaleFactor, 1.2)
    BR_PROPERTY(float, threshold, 0)

    void project(const Template &src, Template &dst) const
    {
        if (src.size() != 1)
            qFatal("Sliding Window only supports templates with 1 mat");

        dst = src;

        const Mat m = src.m();

        int effectiveMaxSize = maxSize;
        if (maxSize < 0)
            effectiveMaxSize = qMax(m.rows, m.cols);

        const float scaleFrom = qMin(winWidth/(float)effectiveMaxSize, winHeight/(float)effectiveMaxSize);
        const float scaleTo = qMax(winWidth/(float)minSize, winHeight/(float)minSize);

        Mat scaledImage; QList<float> confidences;
        for (float scale = scaleFrom; scale < scaleTo + 0.001; scale *= scaleFactor) {
            resize(m, scaledImage, Size(), scale, scale);

            const int step = scale < 1. ? 4 : 8;
            for (int y = 0; y < (scaledImage.rows - winHeight); y += step) {
                for (int x = 0; x < (scaledImage.cols - winWidth); x += step) {
                    Mat window(scaledImage, Rect(x, y, winWidth, winHeight));
                    Template u(src.file, window), t;
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
};

BR_REGISTER(Transform, SlidingWindowTransform)

} // namespace br

#include "imgproc/slidingwindow.moc"
