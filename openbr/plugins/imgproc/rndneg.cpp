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
 * \brief Selects n random regions from a negative image.
 * \author Josh Klontz \cite jklontz
 */
class RndNegTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString label READ get_label WRITE set_label RESET reset_label STORED false)
    Q_PROPERTY(int negVal READ get_negVal WRITE set_negVal RESET reset_negVal STORED false)
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    BR_PROPERTY(QString, label, "Label")
    BR_PROPERTY(int, negVal, 0)
    BR_PROPERTY(int, n, 1)

    void project(const Template &src, Template &dst) const
    {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        RNG &rng = theRNG();
        foreach (const Template &t, src) {
            if (t.file.get<int>(label) != negVal) dst.append(t);
            else {
                for (int i = 0; i < n; i++) {
                    float size = rng.uniform(0.2f, 1.f);
                    float x = rng.uniform(0.f, 1.f-size);
                    float y = rng.uniform(0.f, 1.f-size);

                    Mat m = t.m()(Rect(t.m().cols *x,
                                          t.m().rows *y,
                                          t.m().cols * size,
                                          t.m().rows * size));

                    dst.append(Template(t.file, m));
                }
            }
        }
    }
};

BR_REGISTER(Transform, RndNegTransform)

} // namespace br

#include "imgproc/rndneg.moc"
