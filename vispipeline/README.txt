1. It only works with Java 8. Not wit JDK 7, not with JDK 9 or above, just JDK 8!
2. If the following error appears

java.awt.AWTError: Assistive Technology not found: org.GNOME.Accessibility.AtkWrapper

Edit the file /etc/java-8-openjdk/accessibility.properties and comment out the following line:
assistive_technologies=org.GNOME.Accessibility.AtkWrapper

3. Test with the sample command below, from inside the vispipeline directory:

java -Xms768m -Xmx768m -Djava.library.path=./dlls -Djava.library.path=./components -cp VisPipeline.jar vispipeline.util.CmdLine

Expected output:

VisPipeline Projection Runner
---------------
Usage: vp-run <projection> [arguments]
---------------
The following projections are available:
 * plmp  	 [ Part-Linear Multidimensional Projection (PLMP) ]
 * fastmap  	 [ Fastmap ]
 * idmap  	 [ Interactive Document Map (IDMAP) ]
 * lisomap  	 [ Landmarks Isometric Feature Mapping (Landmarks ISOMAP) ]
 * lsp  	 [ Least Square Projection (LSP) ]
 * pekalska  	 [ Rapid Sammon ]
 * projclus  	 [ Projection by Clustering (ProjClus) ]
 * plsp  	 [ Piecewise Least Square Projection (P-LSP) ]
 * lamp  	 [ LAMP Projection ]

4. Common problems:

- List of available projections empty: usually is due to corruption of some jar files. Is solved by copying the vispipeline folder from another source.

- URL and Class loader exceptions: JDK version is wrong. Check again if using JDK 8.

