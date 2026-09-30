/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import 'jasmine';

import {OrientedDetection as OrientedDetectionProto} from '../../../../framework/formats/oriented_detection_pb';

import {convertFromOrientedDetectionProto} from './oriented_detection_result';

describe('convertFromOrientedDetectionProto()', () => {
  it('transforms custom values', () => {
    const detection = new OrientedDetectionProto();
    detection.setCx(0.5);
    detection.setCy(0.25);
    detection.setWidth(0.4);
    detection.setHeight(0.2);
    detection.setRotation(0.3);
    detection.addScore(0.1);
    detection.addLabelId(1);
    detection.addLabel('ship');
    detection.addDisplayName('Ship');
    detection.setTrackId('track-1');

    const result = convertFromOrientedDetectionProto(detection, 200, 100);

    expect(result.categories).toEqual([{
        score: 0.1,
        index: 1,
        categoryName: 'ship',
        displayName: 'Ship',
    }]);
    expect(result.cx).toBe(100);
    expect(result.cy).toBe(25);
    expect(result.width).toBeCloseTo(86.0971, 3);
    expect(result.height).toBeCloseTo(20.6890, 3);
    expect(result.rotation).toBeCloseTo(0.153452, 5);
    expect(result.trackId).toBe('track-1');
  });

  it('transforms default values', () => {
    const detection = new OrientedDetectionProto();
    detection.addScore(0.2);

    const result = convertFromOrientedDetectionProto(detection, 200, 100);

    expect(result).toEqual({
      categories: [{
        score: 0.2,
        index: -1,
        categoryName: '',
        displayName: '',
      }],
      cx: 0,
      cy: 0,
      width: 0,
      height: 0,
      rotation: 0,
    });
  });
});
