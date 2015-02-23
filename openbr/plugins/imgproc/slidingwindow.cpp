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
    Q_PROPERTY(int step READ get_step WRITE set_step RESET reset_step STORED false)
    Q_PROPERTY(float threshold READ get_threshold WRITE set_threshold RESET reset_threshold STORED false)
    BR_PROPERTY(br::Transform*, transform, NULL)
    BR_PROPERTY(int, winWidth, 24)
    BR_PROPERTY(int, winHeight, 24)
    BR_PROPERTY(int, minSize, 24)
    BR_PROPERTY(int, maxSize, -1)
    BR_PROPERTY(float, scaleFactor, 1.2)
    BR_PROPERTY(int, step, 1)
    BR_PROPERTY(float, threshold, 0.5)

    void project(const Template &src, Template &dst) const
    {
        if (src.size() != 1)
            qFatal("Sliding Window only supports templates with 1 mat");

        dst = src;

        const Mat m = src.first();

        int effectiveMaxSize = maxSize;
        if (maxSize < 0)
            effectiveMaxSize = qMax(m.rows, m.cols);

        QList<float> confidences;
        for (double factor = 1; ; factor *= scaleFactor) {
            Size scaledWindowSize(cvRound(winWidth*factor), cvRound(winHeight*factor));
            Size scaledImageSize(cvRound(m.cols/factor), cvRound(m.rows/factor));

            if (scaledImageSize.width <= winWidth || scaledImageSize.height <= winHeight)
                break;
            if (qMax(scaledWindowSize.width, scaledWindowSize.height) > effectiveMaxSize)
                break;
            if (qMin(scaledWindowSize.width, scaledWindowSize.height) < minSize)
                continue;

            Mat scaledImage(scaledImageSize, CV_32F);
            resize(m, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR);

            for (int y = 0; y < (scaledImage.rows - winHeight); y += step) {
                for (int x = 0; x < (scaledImage.cols - winWidth); x += step) {
                    Template u(src.file, scaledImage(Rect(x, y, winWidth, winHeight))), t;
                    transform->project(u, t);
                    float confidence = t.m().at<float>(0,0);

                    //int cx = qRound(x*factor) + (qRound(winWidth*factor) / 2);
                    //int cy = qRound(y*factor) + (qRound(winHeight*factor) / 2);
                    //conf.at<float>(qMin(m.rows, cy), qMin(m.cols - 1, cx)) += confidence;
                    if (confidence > threshold) {
                        dst.file.appendRect(Rect(qRound(x*factor), qRound(y*factor), qRound(winWidth*factor), qRound(winHeight*factor)));
                        confidences.append(confidence);
                    }
                }
            }
        }
        dst.file.setList<float>("Confidences", confidences);
    }
};

BR_REGISTER(Transform, SlidingWindowTransform)

} // namespace br

#include "imgproc/slidingwindow.moc"
