/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

package bengal

import (
	"os"
	"io/ioutil"
	"encoding/json"
)

func (model *MultioutputMultinomialNB) Serialize() ([]byte, error) {
	return json.Marshal(model)
}

func Deserialize(bytes []byte) (*MultioutputMultinomialNB, error) {
	var model MultioutputMultinomialNB

	if err := json.Unmarshal(bytes, &model); err != nil {
		return nil, err
	} else {
		return &model, nil
	}
}

func (model MultioutputMultinomialNB) SerializeTo(filename string) ([]byte, error) {
	f, err := os.Create(filename)
	if err != nil { return []byte{}, err }
	defer f.Close()

	if b, err := model.Serialize(); err == nil {
		f.Write(b)

		return b, nil
	} else {
		return []byte{}, err
	}
}

func DeserializeFrom(filename string) (*MultioutputMultinomialNB, error) {
	f, err := os.Open(filename)
	if err != nil { return nil, err }
	defer f.Close()

	bytes, err := ioutil.ReadAll(f)
	if err != nil { return nil, err }

	return Deserialize(bytes)
}
