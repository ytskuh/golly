// This file is part of Golly.
// See docs/License.html for the copyright notice.

#include "utils.h"      // for Beep, Warning, FixURLPath
#include "prefs.h"      // for userdir, supplieddir, downloaddir
#include "file.h"       // for OpenFile, UnzipFile, GetURL, DownloadFile, LoadLexiconPattern
#include "control.h"    // for ChangeRule

#import "GollyAppDelegate.h"        // for SwitchToPatternTab, SwitchToHelpTab
#import "InfoViewController.h"      // for ShowTextFile
#import "HelpViewController.h"

@implementation HelpViewController

// -----------------------------------------------------------------------------

- (id)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
    self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
    if (self) {
        self.title = @"Help";
        self.tabBarItem.image = [UIImage imageNamed:@"help.png"];
    }
    return self;
}

// -----------------------------------------------------------------------------

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Release any cached data, images, etc that aren't in use.
}

// -----------------------------------------------------------------------------

- (void)showContentsPage
{
    NSBundle *bundle = [NSBundle mainBundle];
    NSString *filePath = [bundle pathForResource:@"index" ofType:@"html" inDirectory:@"Help"];
    if (filePath) {
        NSURL *fileUrl = [[NSURL fileURLWithPath:filePath] URLByStandardizingPath];
        NSURL *helpDirUrl = [fileUrl URLByDeletingLastPathComponent]; // .../Help/
        [htmlView loadFileURL:fileUrl allowingReadAccessToURL:helpDirUrl];
    } else {
        [htmlView loadHTMLString:@"<html><center><font size=+4 color='red'>Failed to find index.html!</font></center></html>"
                         baseURL:nil];
    }
}

// -----------------------------------------------------------------------------

static WKWebView *globalHtmlView = nil;     // for ShowHelp

- (void)viewDidLoad
{
    [super viewDidLoad];
	
    globalHtmlView = htmlView;
    htmlView.navigationDelegate = self;
    
    // following line will enable zooming content using pinch gestures
    // htmlView.scalesPageToFit = YES;
    // along with a line like this in each .html file's header:
    // <meta name='viewport' content='initial-scale=1.1,maximum-scale=5.0'/>
    // BUT it's simpler to adjust font size using some JavaScript in webView:didFinishNavigation:
    
    backButton.enabled = NO;
    nextButton.enabled = NO;
    
    [self showContentsPage];
}

// -----------------------------------------------------------------------------

- (void)viewWillAppear:(BOOL)animated
{
    [super viewWillAppear:animated];
}

// -----------------------------------------------------------------------------

- (void)viewDidAppear:(BOOL)animated
{
    [super viewDidAppear:animated];
}

// -----------------------------------------------------------------------------

- (void)viewWillDisappear:(BOOL)animated
{
	[super viewWillDisappear:animated];

    [htmlView stopLoading];  // in case the web view is still loading its content
}

// -----------------------------------------------------------------------------

- (void)viewDidDisappear:(BOOL)animated
{
	[super viewDidDisappear:animated];
}

// -----------------------------------------------------------------------------

- (IBAction)doBack:(id)sender
{
    [htmlView goBack];
}

// -----------------------------------------------------------------------------

- (IBAction)doNext:(id)sender
{
    [htmlView goForward];
}

// -----------------------------------------------------------------------------

- (IBAction)doContents:(id)sender
{
    [self showContentsPage];
}

// -----------------------------------------------------------------------------

// WKNavigationDelegate methods:

- (void)webView:(WKWebView *)webView didStartProvisionalNavigation:(WKNavigation *)navigation
{
    [UIApplication sharedApplication].networkActivityIndicatorVisible = YES;
}

// -----------------------------------------------------------------------------

static std::string pageurl;

- (void)webView:(WKWebView *)webView didFinishNavigation:(WKNavigation *)navigation
{
    [UIApplication sharedApplication].networkActivityIndicatorVisible = NO;
    backButton.enabled = htmlView.canGoBack;
    nextButton.enabled = htmlView.canGoForward;
    
    // increase font size
    NSString *jsString = @"document.getElementsByTagName('body')[0].style.webkitTextSizeAdjust= '110%'";
    [htmlView evaluateJavaScript:jsString completionHandler:nil];

    // need URL of this page for relative "get:" links
    // (WKWebView exposes this directly and synchronously, unlike UIWebView, so no
    // need for the old stringByEvaluatingJavaScriptFromString:@"window.location.href" trick)
    NSString *str = webView.URL.absoluteString;
    
    pageurl = [str cStringUsingEncoding:NSUTF8StringEncoding];
    
    // replace each %20 in pageurl with a space (for GetURL)
    for (size_t pos = pageurl.find("%20");
         pos != std::string::npos;
         pos = pageurl.find("%20", pos)) {
        pageurl.replace(pos, 3, " ");
    }
}

// -----------------------------------------------------------------------------

- (void)handleLoadError:(NSError *)error
{
    [UIApplication sharedApplication].networkActivityIndicatorVisible = NO;
    // we can safely ignore -999 errors (eg. when ShowHelp is called before user switches to Help tab)
    if (error.code == NSURLErrorCancelled) return;
    Warning([error.localizedDescription cStringUsingEncoding:NSUTF8StringEncoding]);
}

- (void)webView:(WKWebView *)webView didFailNavigation:(WKNavigation *)navigation withError:(NSError *)error
{
    [self handleLoadError:error];
}

- (void)webView:(WKWebView *)webView didFailProvisionalNavigation:(WKNavigation *)navigation withError:(NSError *)error
{
    [self handleLoadError:error];
}

// -----------------------------------------------------------------------------

- (void)webView:(WKWebView *)webView
    decidePolicyForNavigationAction:(WKNavigationAction *)navigationAction
                    decisionHandler:(void (^)(WKNavigationActionPolicy))decisionHandler
{
    if (navigationAction.navigationType == WKNavigationTypeLinkActivated) {
        NSURL *url = navigationAction.request.URL;
        NSString *link = [url absoluteString];
        // NSLog(@"url = %@", link);
        
        // look for special prefixes used by Golly (and cancel navigation if found)
        if ([link hasPrefix:@"open:"]) {
            // open specified file
            std::string path = [[link substringFromIndex:5] cStringUsingEncoding:NSUTF8StringEncoding];
            FixURLPath(path);
            OpenFile(path.c_str());
            // OpenFile will switch to the appropriate tab
            decisionHandler(WKNavigationActionPolicyCancel);
            return;
        }
        if ([link hasPrefix:@"rule:"]) {
            // switch to specified rule
            std::string newrule = [[link substringFromIndex:5] cStringUsingEncoding:NSUTF8StringEncoding];
            SwitchToPatternTab();
            ChangeRule(newrule);
            decisionHandler(WKNavigationActionPolicyCancel);
            return;
        }
        if ([link hasPrefix:@"lexpatt:"]) {
            // user tapped on pattern in Life Lexicon
            std::string pattern = [[link substringFromIndex:8] cStringUsingEncoding:NSUTF8StringEncoding];
            std::replace(pattern.begin(), pattern.end(), '$', '\n');
            LoadLexiconPattern(pattern);
            decisionHandler(WKNavigationActionPolicyCancel);
            return;
        }
        if ([link hasPrefix:@"edit:"]) {
            std::string path = [[link substringFromIndex:5] cStringUsingEncoding:NSUTF8StringEncoding];
            // convert path to a full path if necessary
            std::string fullpath = path;
            if (path[0] != '/') {
                if (fullpath.find("Patterns/") == 0 || fullpath.find("Rules/") == 0) {
                    // Patterns and Rules directories are inside supplieddir
                    fullpath = supplieddir + fullpath;
                } else {
                    fullpath = userdir + fullpath;
                }
            }
            ShowTextFile(fullpath.c_str());
            decisionHandler(WKNavigationActionPolicyCancel);
            return;
        }
        if ([link hasPrefix:@"get:"]) {
            std::string geturl = [[link substringFromIndex:4] cStringUsingEncoding:NSUTF8StringEncoding];
            // download file specifed in link (possibly relative to a previous full url)
            GetURL(geturl, pageurl);
            decisionHandler(WKNavigationActionPolicyCancel);
            return;
        }
        if ([link hasPrefix:@"unzip:"]) {
            std::string zippath = [[link substringFromIndex:6] cStringUsingEncoding:NSUTF8StringEncoding];
            FixURLPath(zippath);
            std::string entry = zippath.substr(zippath.rfind(':') + 1);
            zippath = zippath.substr(0, zippath.rfind(':'));
            UnzipFile(zippath, entry);
            decisionHandler(WKNavigationActionPolicyCancel);
            return;
        }
        
        // no special prefix, so look for file with .zip/rle/lif/mc extension
        std::string path = [link cStringUsingEncoding:NSUTF8StringEncoding];
        path = path.substr(path.rfind('/')+1);
        size_t dotpos = path.rfind('.');
        std::string ext = "";
        if (dotpos != std::string::npos) ext = path.substr(dotpos+1);
        if ( (IsZipFile(path) ||
                strcasecmp(ext.c_str(),"rle") == 0 ||
                strcasecmp(ext.c_str(),"life") == 0 ||
                strcasecmp(ext.c_str(),"mc") == 0)
                // also check for '?' to avoid opening links like ".../detail?name=foo.zip"
                && path.find('?') == std::string::npos) {
            // download file to downloaddir and open it
            path = downloaddir + path;
            std::string url = [link cStringUsingEncoding:NSUTF8StringEncoding];
            if (DownloadFile(url, path)) {
                OpenFile(path.c_str());
            }
            decisionHandler(WKNavigationActionPolicyCancel);
            return;
        }
    }
    decisionHandler(WKNavigationActionPolicyAllow);
}

@end

// =============================================================================

void ShowHelp(const char* filepath)
{
    SwitchToHelpTab();
    NSURL *fileUrl = [NSURL fileURLWithPath:[NSString stringWithCString:filepath encoding:NSUTF8StringEncoding]];
    NSURLRequest *request = [NSURLRequest requestWithURL:fileUrl];
    [globalHtmlView loadRequest:request];
}
