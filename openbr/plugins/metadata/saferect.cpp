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

namespace br
{

/*!
 * \ingroup transforms
 * \brief Convert values of key_X, key_Y, key_Width, key_Height to a rect.
 * \author Jordan Cheney \cite JordanCheney
 */
class SafeRectTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        QList<QRectF> safeRects;
        foreach (const QRectF rect, src.file.rects()) {
            int x = qMax(rect.x(), 0.);
            int y = qMax(rect.y(), 0.);
            int w = qMin(rect.width(), (double)src.m().cols - x);
            int h = qMin(rect.height(), (double)src.m().rows - y);
            safeRects.append(QRectF(x, y, w, h));
        }

        dst.file.setRects(safeRects);
    }

};

BR_REGISTER(Transform, SafeRectTransform)

} // namespace br

#include "metadata/saferect.moc"
