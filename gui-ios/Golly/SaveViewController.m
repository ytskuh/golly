// This file is part of Golly.
// See docs/License.html for the copyright notice.

#include "writepattern.h"   // for pattern_format, output_compression

#include "utils.h"          // for FileExists
#include "prefs.h"          // for savedir
#include "layer.h"          // for currlayer, etc
#include "file.h"           // for SavePattern, GetBaseName

#import "SaveViewController.h"

@implementation SaveViewController

// -----------------------------------------------------------------------------

static NSString *filetypes[] =          // data for typeTable
{
    // WARNING: code below assumes this ordering
	@"Run Length Encoded (*.rle)",
    @"Compressed RLE (*.rle.gz)",
	@"Macrocell (*.mc)",
	@"Compressed MC (*.mc.gz)"
};
const int NUM_TYPES = sizeof(filetypes) / sizeof(filetypes[0]);

static int currtype = 0;                // current index in typeTable

static bool inSaveTextFile = false;     // SaveTextFile was called?
static const char* initpath;            // path of file being edited
static const char* filedata;            // text to be saved
static std::string fullpath;            // path of file being saved
static InfoViewController* callingVC;   // the view controller that called SaveTextFile

// -----------------------------------------------------------------------------

- (void)checkFileType
{
    std::string filename = [[nameText text] cStringUsingEncoding:NSUTF8StringEncoding];
    if (!filename.empty()) {
        // might need to change currtype to match an explicit extension
        int oldtype = currtype;
        size_t len = filename.length();
        if        (len >= 4 && len - 4 == filename.rfind(".rle")) {
            currtype = 0;
        } else if (len >= 7 && len - 7 == filename.rfind(".rle.gz")) {
            currtype = 1;
        } else if (len >= 3 && len - 3 == filename.rfind(".mc")) {
            currtype = 2;
        } else if (len >= 6 && len - 6 == filename.rfind(".mc.gz")) {
            currtype = 3;
        } else {
            // extension is unknown or not supplied, so append appropriate extension
            size_t dotpos = filename.find('.');
            if (currtype == 0) filename = filename.substr(0,dotpos) + ".rle";
            if (currtype == 1) filename = filename.substr(0,dotpos) + ".rle.gz";
            if (currtype == 2) filename = filename.substr(0,dotpos) + ".mc";
            if (currtype == 3) filename = filename.substr(0,dotpos) + ".mc.gz";
            [nameText setText:[NSString stringWithCString:filename.c_str()
                                                 encoding:NSUTF8StringEncoding]];
        }
        if (currtype != oldtype) {
            [typeTable selectRowAtIndexPath:[NSIndexPath indexPathForRow:currtype inSection:0]
                                   animated:NO
                             scrollPosition:UITableViewScrollPositionNone];
        }
    }
}

// -----------------------------------------------------------------------------

- (void)checkFileName
{
    std::string filename = [[nameText text] cStringUsingEncoding:NSUTF8StringEncoding];
    if (!filename.empty()) {
        // might need to change filename to match currtype
        size_t dotpos = filename.find('.');
        std::string ext = (dotpos == std::string::npos) ? "" : filename.substr(dotpos);
        
        if (currtype == 0 && ext != ".rle")    filename = filename.substr(0,dotpos) + ".rle";
        if (currtype == 1 && ext != ".rle.gz") filename = filename.substr(0,dotpos) + ".rle.gz";
        if (currtype == 2 && ext != ".mc")     filename = filename.substr(0,dotpos) + ".mc";
        if (currtype == 3 && ext != ".mc.gz")  filename = filename.substr(0,dotpos) + ".mc.gz";
        
        [nameText setText:[NSString stringWithCString:filename.c_str()
                                             encoding:NSUTF8StringEncoding]];
    }
}

// -----------------------------------------------------------------------------

- (id)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
    self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
    if (self) {
        self.title = @"Save";
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

- (void)viewDidLoad
{
    [super viewDidLoad];
    // Do any additional setup after loading the view from its nib.
}

// -----------------------------------------------------------------------------

- (void)viewWillAppear:(BOOL)animated
{
    [super viewWillAppear:animated];
    
    if (inSaveTextFile) {
        // user is saving a pattern (in Documents/Saved or Documents/Downloads)
        // or a .rule file (in Documents/Rules)
        
        std::string filename = GetBaseName(initpath);
        [nameText setText:[NSString stringWithCString:filename.c_str() encoding:NSUTF8StringEncoding]];
        typeTable.hidden = YES;
        
        // change labels
        [topLabel setText:@"Save file as:"];
        std::string filepath = initpath;
        if (filepath.find("Documents/Rules/") != std::string::npos) {
            [botLabel setText:@"File location: Documents/Rules/"];
        } else if (filepath.find("Documents/Downloads/") != std::string::npos) {
            [botLabel setText:@"File location: Documents/Downloads/"];
        } else {
            [botLabel setText:@"File location: Documents/Saved/"];
        }

    } else {
        // user is saving current pattern via Pattern tab's Save button

        // [nameText setText:[NSString stringWithCString:currlayer->currname.c_str() encoding:NSUTF8StringEncoding]];
        // probably nicer not to show current name
        [nameText setText:@""];

        // init file type
        if (currlayer->algo->hyperCapable()) {
            // RLE is allowed but macrocell format is better
            if (currtype < 2) currtype = 2;
        } else {
            // algo doesn't support macrocell format
            if (currtype > 1) currtype = 0;
        }
        [typeTable selectRowAtIndexPath:[NSIndexPath indexPathForRow:currtype inSection:0]
                               animated:NO
                         scrollPosition:UITableViewScrollPositionNone];
        
        [self checkFileName];
    }
    
    // show keyboard immediately
    [nameText becomeFirstResponder];
}

// -----------------------------------------------------------------------------

- (void)viewWillDisappear:(BOOL)animated
{
	[super viewWillDisappear:animated];
    
    inSaveTextFile = false;
}

// -----------------------------------------------------------------------------

- (IBAction)doCancel:(id)sender
{
    [self dismissViewControllerAnimated:YES completion:nil];
}

// -----------------------------------------------------------------------------

- (void)CompleteSave
{
    // finish the save
    if (inSaveTextFile) {
        FILE* f = fopen(fullpath.c_str(), "w");
        if (f) {
            if (fputs(filedata, f) == EOF) {
                fclose(f);
                Warning("Could not write to file!");
                return;
            }
        } else {
            Warning("Could not create file!");
            return;
        }
        fclose(f);
    
        [self dismissViewControllerAnimated:YES completion:nil];
    
        // tell caller that the save was a success
        [callingVC saveSucceded:fullpath.c_str()];

    } else {
        // dismiss modal view first in case SavePattern calls BeginProgress
        [self dismissViewControllerAnimated:YES completion:nil];

        pattern_format format = currtype < 2 ? XRLE_format : MC_format;
        output_compression compression = currtype % 2 == 0 ? no_compression : gzip_compression;
        SavePattern(fullpath, format, compression);
    }
}

// -----------------------------------------------------------------------------

// using YesNo freezes the app (probably because the Save view is modal) so use this instead

- (void)ReplaceQuery
{
    UIAlertController *alert = [UIAlertController
        alertControllerWithTitle:@"Warning"
        message:@"A file with that name already exists.\nDo you want to replace that file?"
        preferredStyle:(UIAlertControllerStyleAlert)];
    
    UIAlertAction *cancelAction = [UIAlertAction
        actionWithTitle:@"No"
        style:UIAlertActionStyleCancel
        handler:^(UIAlertAction * _Nonnull action) {
            // doing nothing will dismiss the view
        }];
    [alert addAction:cancelAction];

    UIAlertAction *okAction = [UIAlertAction
        actionWithTitle:@"Yes"
        style:UIAlertActionStyleDefault
        handler:^(UIAlertAction * _Nonnull action) {
            // handle Yes response here
            [self CompleteSave];
        }];
    [alert addAction:okAction];

    Beep();
    [self presentViewController:alert animated:YES completion:nil];
}

// -----------------------------------------------------------------------------

- (IBAction)doSave:(id)sender
{
    // clear first responder if necessary (ie. remove keyboard)
    [self.view endEditing:YES];
    
    std::string filename = [[nameText text] cStringUsingEncoding:NSUTF8StringEncoding];
    if (filename.empty()) {
        Beep();
        [nameText becomeFirstResponder];
        return;
    }
    
    if (inSaveTextFile) {
        // user is saving a text file (pattern or .rule file)
        std::string initname = GetBaseName(initpath);
        std::string dir = initpath;
        dir = dir.substr(0,dir.rfind('/')+1);
        
        // prevent Documents/Rules/* being saved as something other than a .rule file
        if (EndsWith(dir,"Documents/Rules/") && !EndsWith(filename,".rule")) {
            Warning("Files in Documents/Rules/ must have a .rule extension.");
            return;
        }
        
        fullpath = dir + filename;
        if (initname != filename && FileExists(fullpath)) {
            // ask user if it's ok to replace an existing file that's not the same as the given file
            [self ReplaceQuery];
            return;
        }
    
    } else {
        // user is saving current pattern in Documents/Saved/ via Pattern tab's Save button
        fullpath = savedir + filename;
        if (FileExists(fullpath)) {
            // ask user if it's ok to replace an existing file
            [self ReplaceQuery];
            return;
        }
    }
    
    [self CompleteSave];
}

// -----------------------------------------------------------------------------

// UITextFieldDelegate methods:

- (void)textFieldDidEndEditing:(UITextField *)tf
{
    // called when rule editing has ended (ie. keyboard disappears)
    if (!inSaveTextFile) [self checkFileType];
}

- (BOOL)textFieldShouldReturn:(UITextField *)tf
{
    // called when user hits Done button, so remove keyboard
    // (note that textFieldDidEndEditing will then be called)
    [tf resignFirstResponder];
    return YES;
}

- (BOOL)disablesAutomaticKeyboardDismissal
{
    // this allows keyboard to be dismissed if modal view uses UIModalPresentationFormSheet
    return NO;
}

// -----------------------------------------------------------------------------

// UITableViewDelegate and UITableViewDataSource methods:

- (NSInteger)numberOfSectionsInTableView:(UITableView *)TableView
{
	return 1;
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)component
{
    if (!currlayer->algo->hyperCapable()) {
        // algo doesn't support macrocell format
        return 2;
    } else {
        return NUM_TYPES;
    }
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath
{
    // check for a reusable cell first and use that if it exists
    UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"UITableViewCell"];
    
    // if there is no reusable cell of this type, create a new one
    if (!cell) cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleDefault
                                             reuseIdentifier:@"UITableViewCell"];
    
    [[cell textLabel] setText:filetypes[[indexPath row]]];
	return cell;
}

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath
{
    // change selected file type
    int newtype = (int)[indexPath row];
    if (newtype < 0 || newtype >= NUM_TYPES) {
        Warning("Bug: unexpected row!");
    } else {
        currtype = newtype;
        [self checkFileName];
    }
}

@end

// =============================================================================

void SaveTextFile(const char* filepath, const char* contents, InfoViewController* currentView)
{
    inSaveTextFile = true;      
    initpath = filepath;
    filedata = contents;
    callingVC = currentView;
    
    SaveViewController *modalSaveController = [[SaveViewController alloc] initWithNibName:nil bundle:nil];
    
    [modalSaveController setModalPresentationStyle:UIModalPresentationFormSheet];
    [currentView presentViewController:modalSaveController animated:YES completion:nil];
    
    modalSaveController = nil;
    
    // cannot reset inSaveTextFile here (it must be done in viewWillDisappear)
    // because doSave is called AFTER SaveTextFile finishes
}
