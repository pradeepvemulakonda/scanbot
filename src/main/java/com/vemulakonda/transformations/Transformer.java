package com.vemulakonda.transformations;


import com.vemulakonda.common.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

import static org.opencv.core.Core.NORM_MINMAX;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgproc.Imgproc.THRESH_TRUNC;
import static org.opencv.imgproc.Imgproc.medianBlur;
import static org.opencv.imgproc.Imgproc.morphologyEx;

public class Transformer {

    public static Builder transform(Mat image) {
        Builder builder = new Builder(image);
        return builder;
    }

    public static class Builder {
        private Mat source;
        private Mat destination;

        Builder(Mat source) {
            this.source = source;
        }

        public Builder toGraySacle() {
            destination = Utils.toGrayScale(source);
            return this;
        }

        public Builder removeNoise() {
            Mat dest = destination != null ? destination : source;
            Photo.fastNlMeansDenoising(dest, dest);
            return this;
        }

        public Builder morphOpen(int siz) {
            morph(5, Imgproc.MORPH_OPEN);
            return this;
        }

        public Builder morphClose(int siz) {
            morph(5, Imgproc.MORPH_CLOSE);
            return this;
        }

        public Builder erode(int size) {
            Imgproc.erode(getSrc(), getDest(), getKernel(size, size));
            return this;
        }

        public Builder dilate(int size) {
            Imgproc.dilate(getSrc(), getDest(), getKernel(size, size));
            return this;
        }

        public Builder doMedianBlur(int size) {
            medianBlur(getSrc(), getDest(), 21);
            return this;
        }

        public Builder diffWith(Mat image) {
            Core.absdiff(image, getDest(), getDest());
            return this;
        }

        public Builder inverse() {
            Core.bitwise_not(getSrc(), getDest());
            return this;
        }

        public Builder normalize() {
            Core.normalize(getSrc(), getDest(), 0, 255, NORM_MINMAX, CV_8UC1);
            return this;
        }

        public Builder threshold(int threshold, int maxValue) {
            Imgproc.threshold(getSrc(), getDest(), threshold, maxValue, THRESH_TRUNC);
            return this;
        }

        public Builder resize(int width, int height) {
            Imgproc.resize(getSrc(), getDest(), new Size(width, height), 0, 0, Imgproc.INTER_CUBIC);
            return this;
        }

        public Builder addWeight(double alpha, double beta, double gamma) {
            Core.addWeighted(getSrc(), alpha, getDest(), beta, gamma, getDest());
            return this;
        }






        public Mat build() {
            return destination;
        }

        private Builder morph(int size, int morphType) {
            morphologyEx(getSrc(), getDest(), morphType, getKernel(size, size));
            return this;
        }



        private Mat getKernel(int rows, int columns) {
            return Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(rows, columns));
        }

        private Mat getSrc() {
            return destination != null ? destination : source;
        }

        private Mat getDest() {
            destination =  destination == null ? getNewDest() : destination;
            return destination;
        }

        private Mat getNewDest() {
            return new Mat();
        }
    }

}
