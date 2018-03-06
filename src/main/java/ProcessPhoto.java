import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

import static org.opencv.core.Core.NORM_MINMAX;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgproc.Imgproc.*;

/**
 * Created by Harita on 4/3/18.
 */
public class ProcessPhoto {

    public static void main(String[] args) {
        int dilation_size = 1;
        System.load("/Users/Harita/IdeaProjects/scanbot/libs/libopencv_java341.dylib");
        Mat source = Imgcodecs.imread("/Users/Harita/IdeaProjects/scanbot/src/main/resources/exmaple2.jpg");
        Mat dest = new Mat();
        Mat dest2 = new Mat();
        Photo.fastNlMeansDenoising(source, dest);
        Imgproc.cvtColor(dest, dest2, Imgproc.COLOR_RGB2GRAY);
        Mat element1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(9, 9));
        morphologyEx(dest2, dest2, Imgproc.MORPH_CLOSE, element1);
        medianBlur(dest2, dest2,  21);
        Mat original = new Mat();
        Imgproc.cvtColor(source, original, Imgproc.COLOR_BGR2GRAY);
        Core.absdiff(original, dest2, dest2);
        Core.bitwise_not(dest2, dest2);
        Mat normalImage = new Mat();
        dest2.copyTo(normalImage);
        Core.normalize(dest2, normalImage, 0, 255, NORM_MINMAX, CV_8UC1);
        Imgproc.threshold(normalImage, normalImage, 230, 0, THRESH_TRUNC);
        Core.normalize(normalImage, normalImage, 0, 255, NORM_MINMAX, CV_8UC1);
        Imgcodecs.imwrite("photo.jpeg", normalImage);
    }
}
