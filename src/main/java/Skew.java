import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.opencv.imgproc.Imgproc.INTER_CUBIC;


public class Skew {


    public double getskewAngle(Mat src) {
        Mat dst = new Mat(src.rows(), src.cols(), src.type());
        Mat cdst = new Mat();
        Imgproc.GaussianBlur(src, dst, new Size(5, 5), 1, 1);

        Core.addWeighted(dst, 1.5, dst, -0.5, 0, dst);
        Imgproc.cvtColor(src, dst, Imgproc.COLOR_BGR2GRAY);
        final List<MatOfPoint> points = new ArrayList<>();
        final Mat hierarchy = new Mat();
        p(dst);
        Imgproc.findContours(dst, points, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        Random r = new Random();
        for (int i = 0; i < points.size(); i++) {
            Imgproc.drawContours(dst, points, i, new Scalar(r.nextInt(255), r.nextInt(255), r.nextInt(255)), -1);
        }

        Imgcodecs.imwrite("2.jpg", dst);
        // Edge detection
        Imgproc.Canny(dst, dst, 50, 200, 3, false);
        Imgproc.cvtColor(dst, cdst, Imgproc.COLOR_GRAY2BGR);
        Imgcodecs.imwrite("gray.jpg", cdst);
        Mat lines = new Mat(); // will hold the results of the detection
        Imgproc.HoughLines(dst, lines, 1, Math.PI / 180, 60); // runs the actual detection
        double angle = 0;
        System.out.println(lines.size());
        for (int x = 0; x < lines.rows(); x++) {
            double rho = lines.get(x, 0)[0],
                    theta = lines.get(x, 0)[1];
            double a = Math.cos(theta), b = Math.sin(theta);
            double x0 = a * rho, y0 = b * rho;
            Point pt1 = new Point(Math.round(x0 + 1000 * (-b)), Math.round(y0 + 1000 * (a)));
            Point pt2 = new Point(Math.round(x0 - 1000 * (-b)), Math.round(y0 - 1000 * (a)));
            Imgproc.line(cdst, pt1, pt2, new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
            angle += Math.atan2((float) (pt2.x - pt1.x), (float) (pt2.y - pt1.y));
        }


        Imgcodecs.imwrite("output.jpg", cdst);

        angle = angle / lines.rows();
        p("--->" + angle);

        angle = Math.toDegrees(angle);
        System.out.println(angle);
        return angle;
    }

    private void p(Object obj) {
        System.out.println(obj);
    }


    public static void main(String[] args) {
        System.load("libs/libopencv_java341.dylib");
        Mat src = Imgcodecs.imread("src/main/resources/example2.jpg");
        Skew skew = new Skew();
        double skewAngle = skew.getskewAngle(src);
        skewAngle = 90 - skewAngle;
        Mat dst = new Mat(src.rows(), src.cols(), CvType.CV_8UC1);
        Core.bitwise_not(src, dst);

        Point center = new Point(src.width() / 2, src.height() / 2);

        Mat rotated = new Mat();
        Mat m = Imgproc.getRotationMatrix2D(center, skewAngle, 1.0);
        Imgproc.warpAffine(dst, rotated, m, dst.size(), INTER_CUBIC);
        Core.bitwise_not(rotated, rotated);
        Imgcodecs.imwrite("new.jpeg", rotated);
    }

}
