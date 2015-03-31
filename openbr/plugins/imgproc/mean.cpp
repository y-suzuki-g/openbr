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

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Computes the mean of a set of templates.
 * \note Suitable for visualization only as it sets every projected template to the mean template.
 * \author Scott Klum \cite sklum
 */
class MeanTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Mat mean;
    int count;

public:
    MeanTransform() : TimeVaryingTransform(false, false) {}

private:
    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        if (mean.empty()) mean = Mat::zeros(src.first().m().rows, src.first().m().cols, CV_32F);

        (void) dst;

        foreach (const Template &t, src) {
            Mat converted;
            t.m().convertTo(converted, CV_32F);
            mean += converted;
        }

        count += src.size();
    }

    void finalize(TemplateList &output)
    {
        mean /= count;
        Template t(File(), mean);
        output.clear();
        output.append(t);
    }
};

BR_REGISTER(Transform, MeanTransform)

} // namespace br

#include "imgproc/mean.moc"
