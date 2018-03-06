import com.vemulakonda.common.Utils;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import static org.opencv.imgproc.Imgproc.INTER_CUBIC;

/**
 * Created by Harita on 4/3/18.
 */
public class Erode {

    public static void main(String[] args) {
        System.load("lib/libopencv_java341.dylib");
        Mat source = Imgcodecs.imread("src/main/resources/example2.jpg");

        Mat eroded = new Mat(source.rows(),source.cols(),source.type());
        Mat dilated = new Mat(source.rows(),source.cols(),source.type());;
        Mat intermediate = new Mat(source.rows(),source.cols(),source.type());

        int erosion_size = 3;
        int dilation_size = 7;

        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2*erosion_size + 1, 2*erosion_size+1));
        Imgproc.erode(source, eroded, element);
        Imgcodecs.imwrite("erode.jpeg", eroded);


        Mat element1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new  Size(2*dilation_size + 1, 2*dilation_size+1));
        Imgproc.dilate(eroded, dilated, element1);
        Imgcodecs.imwrite("dilate.jpeg", dilated);


        Imgproc.Canny(dilated, intermediate, 50, 200, 3, false);
        Imgcodecs.imwrite("canny.jpeg", intermediate);
        double skewAngle = Utils.getSkewAngle(intermediate);
        skewAngle = 90 - skewAngle;
        System.out.println(skewAngle);
        Point center = new Point(source.width()/2, source.height()/2);

        Mat rotated = new Mat();
        Mat blurred = new Mat();
        Mat m = Imgproc.getRotationMatrix2D(center, skewAngle, 1.0);
        Imgproc.warpAffine( source, rotated, m, source.size(), INTER_CUBIC);
        //Core.bitwise_not(rotated, rotated);
        Imgproc.resize(rotated, rotated, new Size(1500, 2100), 0, 0, Imgproc.INTER_CUBIC);
        Imgproc.GaussianBlur(rotated, blurred, new Size(3, 3), 0);
        Core.bitwise_not(blurred, blurred);
        Core.addWeighted(rotated, 1.2, blurred, -0.8, 0, blurred);

        Imgcodecs.imwrite("new.jpeg", blurred);
    }


}
