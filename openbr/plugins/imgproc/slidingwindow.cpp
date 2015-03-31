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

    Q_PROPERTY(br::Classifier* classifier READ get_classifier WRITE set_classifier RESET reset_classifier STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(int maxSize READ get_maxSize WRITE set_maxSize RESET reset_maxSize STORED false)
    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(float threshold READ get_threshold WRITE set_threshold RESET reset_threshold STORED false)
    BR_PROPERTY(br::Classifier*, classifier, NULL)
    BR_PROPERTY(int, minSize, 24)
    BR_PROPERTY(int, maxSize, -1)
    BR_PROPERTY(float, scaleFactor, 1.2)
    BR_PROPERTY(float, threshold, 0.0)

    void train(const TemplateList &data)
    {
        QList<Mat> images;
        foreach (const Mat &m, data.data())
            images.append(classifier->preprocess(m));

        classifier->train(images, File::get<float>(data, "Label"));
    }

    void project(const Template &src, Template &dst) const
    {
        if (src.size() != 1)
            qFatal("Sliding Window only supports templates with 1 mat");

        dst = src;

        Size winSize = classifier->windowSize();

        const Mat m = classifier->preprocess(src);

        int effectiveMaxSize = maxSize;
        if (maxSize < 0)
            effectiveMaxSize = qMax(m.rows, m.cols);

        const float scaleFrom = qMin(winSize.width/(float)effectiveMaxSize, winSize.height/(float)effectiveMaxSize);
        const float scaleTo = qMax(winSize.width/(float)minSize, winSize.height/(float)minSize);

        Mat scaledImage; QList<float> confidences;
        for (float scale = scaleFrom; scale < scaleTo + 0.001; scale *= scaleFactor) {
            qDebug("scaledImage: %dx%d", (int)scale*m.rows, (int)scale*m.cols);
            resize(m, scaledImage, Size(), scale, scale, CV_INTER_LINEAR);

            const int step = scale < 1. ? 1 : 2;
            for (int y = 0; y < (scaledImage.rows - winSize.width); y += step) {
                for (int x = 0; x < (scaledImage.cols - winSize.height); x += step) {
                    Mat window(scaledImage, Rect(Point(x, y), winSize));
                    float confidence = classifier->classify(window);
                    if (confidence > threshold) {
                        dst.file.appendRect(Rect(qRound(x/scale), qRound(y/scale), qRound(winSize.width/scale), qRound(winSize.height/scale)));
                        confidences.append(confidence);
                    }
                }
            }
            dst.file.setList<float>("Confidences", confidences);
        }
    }

    void store(QDataStream &stream) const
    {
        classifier->store(stream);
    }

    void load(QDataStream &stream)
    {
        classifier->load(stream);
    }
};

BR_REGISTER(Transform, SlidingWindowTransform)

} // namespace br

#include "imgproc/slidingwindow.moc"
