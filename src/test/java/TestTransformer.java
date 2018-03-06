import com.vemulakonda.common.Utils;
import com.vemulakonda.transformations.Transformer;
import org.junit.Test;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Created by Harita on 5/3/18.
 */

public class TestTransformer {

    @Test
    public void testTransformer() {
        System.load(Paths.get("libs/libopencv_java341.dylib").toAbsolutePath().toString());
        Mat source = Imgcodecs.imread("src/main/resources/example2.jpg");
        Mat dest = null;
        dest = Utils.toGrayScale(source);

        Mat inverse = Transformer.transform(source)
                .toGraySacle()
                .removeNoise()
                .dilate(10)
                .erode(5)
                .diffWith(dest)
                .resize(600, 800)
                .dilate(1)
                .removeNoise()
                .normalize()
                .addWeight(1.2, -0.8, 0)
                .normalize()
                .inverse()
                .removeNoise()
                .normalize()
                .threshold(245, 255)
                .build();
        System.out.println(inverse);
        Utils.writeToFile("test.jpeg", inverse);
    }
}
